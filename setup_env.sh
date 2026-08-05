#!/bin/bash
# Creates the conda environment on the login node. Run once:
#
#   ssh atos
#   cd ~/galaxy-morphology
#   bash setup_env.sh
#
# The compute nodes have H100 cards (sm_90), so PyTorch comes from the cu121
# wheel index. Everything else is in requirements.txt.
set -e

ENV_NAME=${ENV_NAME:-galaxy}
PY_VER=3.10
TORCH_INDEX=https://download.pytorch.org/whl/cu121

source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "environment '$ENV_NAME' already exists, updating it"
    conda activate "$ENV_NAME"
else
    conda create -y -n "$ENV_NAME" python=$PY_VER
    conda activate "$ENV_NAME"
fi

python -m pip install --upgrade pip
pip install torch torchvision --index-url $TORCH_INDEX
pip install -r requirements.txt

echo
echo "installed:"
python - <<'PY'
import torch, timm
print("  torch", torch.__version__, "| timm", timm.__version__)
print("  cuda visible here:", torch.cuda.is_available(), "(False on the login node is normal)")
PY

# The pretrained weights are pulled from the Hugging Face hub the first time a
# backbone is built. Compute nodes usually have no outbound network, so warm the
# cache now, from the login node.
echo
echo "pre-downloading pretrained weights..."
python -m src.models --download

echo
echo "done. Remember to export GZM_DATA and GZM_WORK (see README)."
