from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

from openpi.policies.policy import Policy as _PolicyBase
from openpi.policies.policy import PolicyRecorder # for completeness


class Policy(_PolicyBase):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        super().__init__(model, rng=rng, transforms=transforms, output_transforms=output_transforms, sample_kwargs=sample_kwargs, metadata=metadata, pytorch_device=pytorch_device, is_pytorch=is_pytorch)
        if self._is_pytorch_model:
            self._sample_actions = model.sample_actions_with_features # NOTE: go to the new pi0.py
        else:
            self._sample_actions = nnx_utils.module_jit(model.sample_actions_with_features)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        return self.infer_batch(obs, noise=noise)
        
    def infer_batch(self, obs: dict, *, noise: np.ndarray | None = None, nbatch=1) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        nbatch = int(obs.pop("__nbatch__", 1))

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(
                lambda x: jnp.repeat(jnp.asarray(x)[np.newaxis, ...], nbatch, axis=0), inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device).unsqueeze(0).repeat(
                    nbatch, *([1] * np.array(x).ndim)
                ),
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        out_vla = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs) # XXX
        outputs = {
            "state": inputs["state"],
            "actions": out_vla[0],
        }
        model_time = time.monotonic() - start_time

        ### XXX
        # if self._is_pytorch_model:
        #     outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        # else:
        #     outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        # outputs = self._output_transform(outputs)

        action_list = []
        if self._is_pytorch_model:
            output_i = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            output_i = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        action_list.append(self._output_transform(output_i)["actions"])
        for i in range(outputs["actions"].shape[0]-1):
            if self._is_pytorch_model:
                output_i = jax.tree.map(lambda x: np.asarray(x[i+1, ...].detach().cpu()), outputs)
            else:
                output_i = jax.tree.map(lambda x: np.asarray(x[i+1, ...]), outputs)
            output_i = self._output_transform(output_i)
            action_list.append(output_i["actions"])
        outputs = {
            "actions": np.stack(action_list, axis=0),
            "policy_timing": {
                "infer_ms": model_time * 1000,
            },
        }
        
        vlm = out_vla[1]
        if self._is_pytorch_model:
            vlm = vlm[0, ...].detach().to(torch.float32).cpu().numpy()
        else:
            vlm = np.asarray(jax.device_get(vlm[0, ...]), dtype=np.float32) # device_get forces transfer from GPU/TPU to host
        outputs["vlm_fea"] = vlm

        return outputs



