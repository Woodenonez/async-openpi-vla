## Implementation
This repository is based on the [OpenPI](https://github.com/Physical-Intelligence/openpi) project and is **NOT** self-contained. The basic implementation is to introduce an asynchronous inference interface with the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark.

The currently used openpi commit is [981483d](https://github.com/Physical-Intelligence/openpi/tree/981483dca0fd9acba698fea00aa6e52d56a66c58). Later commits may be updated in the future and are not guaranteed to be compatible.

## Dependencies
Apart from the dependencies of the original OpenPI project, this repository also depends on the LIBERO benchmark. Extra dependencies are:
- bddl>=3.6.0
- robosuite==1.5.1
- easydict>=1.13

## Preparation
- Clone the original OpenPI repository (better to use the exact commit mentioned above), and copy the extra content in this repository to the OpenPI repository. Also update the `pyproject.toml` file according to the changes in this repository.

- It is recommended to use `uv` to manage the Python environment. You can create a new environment and install the dependencies using the following commands:
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```
To install Python with the specified version (e.g., 3.11), 
```bash
uv python install 3.11
uv venv --python 3.11
```

## Modification
1. Modify the `serve_policy.py` file (`scripts/serve_policy.py`) to restrain the memory occupation of JAX. 
2. Modify the `policy.py` file (`src/openpi/policy/policy.py`) to add the optional functions (see below). If any of the optional functions is needed, this should be modified as well; otherwise, it can be left unchanged.
### Batch inference (optional)
This is optional if you don't need batch inference.
1. Modify the `websocket_client_policy.py` file (`packages/openpi-client/src/openpi_client/websocket_client_policy.py`) to add batch inference support. 
2. Modify the `base_policy.py` file (`packages/openpi-client/src/openpi_client/base_policy.py`) to enable extra arguments in the `infer` method.
### Feature output (optional)
This is optional if you don't need the VLM feature output.
1. Modify the `pi0.py` file (`src/openpi/models/pi0.py`) to add the feature output in the `sample_actions` method.

### Others
There are other scripts that are mainly used to facilitate the usage, and are not necessary for the main functionality. 

## Usage
### Inference
To run the policy server, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8888 policy:checkpoint   --policy.config=pi05_libero   --policy.dir=/some_user/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```
Change the corresponding arguments according to your needs. 
After the server is running, you can use the websocket client to connect to the server and perform inference.
```bash
python client_inference/client.py --port 8888
```
Optional arguments can be found in `client_inference/prepare_client.py`. 
1. Sychronous inference: `client.py`
2. Asychronous inference: `client_rt.py`