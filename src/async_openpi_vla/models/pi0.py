import logging

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

from openpi.models.pi0 import make_attn_mask, Pi0 as _Pi0Base

logger = logging.getLogger("openpi")


class Pi0(_Pi0Base):
    _extended = True

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
    
    def sample_actions_with_features(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
            # noise = jax.numpy.zeros((batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # ---- probe VLM feature shape (concrete) for loop carry init ----
        suffix_tokens0, suffix_mask0, suffix_ar_mask0, adarms_cond0 = self.embed_suffix(
            observation, noise, jnp.ones((batch_size,), dtype=noise.dtype)  # time=1.0 broadcasted
        )
        suffix_attn_mask0 = make_attn_mask(suffix_mask0, suffix_ar_mask0)
        prefix_attn_mask0 = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens0.shape[1])
        full_attn_mask0 = jnp.concatenate([prefix_attn_mask0, suffix_attn_mask0], axis=-1)
        positions0 = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask0, axis=-1) - 1

        (_, suffix_out0), _ = self.PaliGemma.llm(
            [None, suffix_tokens0],
            mask=full_attn_mask0,
            positions=positions0,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond0],
        )

        # This is exactly what you want to export later:
        vlm_feat0 = suffix_out0[:, -self.action_horizon:]
        last_vlm_init = jnp.zeros_like(vlm_feat0) 

        def step(carry):
            x_t, time, last_vlm_fea = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            vlm_fea = suffix_out[:, -self.action_horizon:]
            v_t = self.action_out_proj(vlm_fea)

            return x_t + dt * v_t, time + dt, vlm_fea

        def cond(carry):
            x_t, time, last_vlm_fea = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, vlm_fea = jax.lax.while_loop(cond, step, (noise, 1.0, last_vlm_init))
        return x_0, vlm_fea
