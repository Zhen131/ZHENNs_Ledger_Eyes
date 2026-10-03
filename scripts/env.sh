# Shared environment for the helper scripts. Source it, do not run it.
# Keeps every download and cache under data/ and switches off anything
# that would send data anywhere other than the package index, Hugging Face
# or the NLTK data server.

if [ -z "${BASH_SOURCE[0]:-}" ]; then
    echo "scripts/env.sh must be sourced from bash" >&2
    return 1
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR
export EYES_PYTHON="${EYES_PYTHON:-$REPO_DIR/.venv/bin/python}"

mkdir -p "$REPO_DIR/data/hf_home" "$REPO_DIR/data/nltk_data" "$REPO_DIR/data/pip-cache" "$REPO_DIR/data/mpl-config"

# Hugging Face: cache under data/, never send a stored token, no telemetry,
# plain HTTPS downloads instead of the optional Xet client.
export HF_HOME="$REPO_DIR/data/hf_home"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export DO_NOT_TRACK=1

# NLTK resources go to data/nltk_data.
export NLTK_DATA="$REPO_DIR/data/nltk_data"

# pip: cache under data/, no self version check.
export PIP_CACHE_DIR="$REPO_DIR/data/pip-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Gradio (the notebook's panel): no usage statistics and no version checks; temporary files under data/.
export GRADIO_ANALYTICS_ENABLED=False
mkdir -p "$REPO_DIR/data/gradio-tmp"
export GRADIO_TEMP_DIR="$REPO_DIR/data/gradio-tmp"

# Matplotlib font cache under data/.
export MPLCONFIGDIR="$REPO_DIR/data/mpl-config"

# Keras settings file (keras.json) under data/, so a ~/.keras on this machine is neither read nor written.
mkdir -p "$REPO_DIR/data/keras"
export KERAS_HOME="$REPO_DIR/data/keras"

# IPython profile and Jupyter runtime files under data/.
mkdir -p "$REPO_DIR/data/ipython" "$REPO_DIR/data/jupyter-runtime"
export IPYTHONDIR="$REPO_DIR/data/ipython"
export JUPYTER_RUNTIME_DIR="$REPO_DIR/data/jupyter-runtime"

# Jupyter's own data and settings (notebook signatures, migration marker) under data/.
# Kernel specs are still found in the virtual environment.
mkdir -p "$REPO_DIR/data/jupyter-data" "$REPO_DIR/data/jupyter-config"
export JUPYTER_DATA_DIR="$REPO_DIR/data/jupyter-data"
export JUPYTER_CONFIG_DIR="$REPO_DIR/data/jupyter-config"
