# Asynchronous OpenPI Simulation

Run the pi VLA model in the LIBERO environment asynchronously, with support of batch inference and feature output for testing.

## Implementation

This repository is based on the [OpenPI](https://github.com/Physical-Intelligence/openpi) project. The basic implementation is to introduce an asynchronous inference interface with the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark.

The currently used openpi commit is [15a9616](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac). Later commits may be updated in the future and are not guaranteed to be compatible.

## Repository layout

| Path | Type | Description |
| --- | --- | --- |
| `extern/openpi` | Git submodule | Upstream [OpenPI](https://github.com/Physical-Intelligence/openpi), pinned to commit [15a9616](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac) |
| `extern/libero` | Vendored (modified) | Modified [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) kept in-tree for robosuite compatibility |
| `src/async-openpi-vla/` | Patches | For asynchronous inference and feature output, some files wrap the original OpenPI files. |
| `scripts/`, `packages/`, etc. | Patches | Others |

## Preparation

### Clone

```bash
git clone --recurse-submodules https://github.com/Woodenonez/async-openpi-vla.git
cd async-openpi-vla
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

This initializes `extern/openpi` and also pulls OpenPI's own submodules (`third_party/aloha`, `third_party/libero`). For evaluation, use the modified LIBERO in `extern/libero` instead of `extern/openpi/third_party/libero`.

### Python environment

It is recommended to use `uv` to manage the Python environment. Create a new environment and install dependencies:

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

Add extra functions to support batch inference and VLM feature output.

## Usage

### LIBERO Path

If the LIBERO path is not set, you can set it by running the following command:

```bash
uv run python -c "import libero; libero.set_libero_default_path()"
```

### Inference

To run the policy server, use the following command (replace the path with your own):

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port 8888 policy:checkpoint   --policy.config=pi05_libero   --policy.dir=/some_user/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

Change the corresponding arguments according to your needs. After the server is running, you can use the WebSocket client to connect to the server and perform inference (synchronous or asynchronous with the `rt` flag; For more, check Args in `client_inference/prepare_client.py`).

```bash
python scripts/client.py --port 8888 --rt
```

## License

This repository combines code from multiple sources under different licenses:

| Component | License | File |
| --- | --- | --- |
| async-openpi-vla (original contributions) | MIT License | [LICENSE](LICENSE) |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) (commit [15a9616](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac)) | Apache License 2.0 | [LICENSE_OPENPI](LICENSE_OPENPI) |
| Gemma model components | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt) |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark (modified) | MIT License | [extern/libero/LICENSE](extern/libero/LICENSE) |

See also [NOTICE](NOTICE) for third-party attribution.
