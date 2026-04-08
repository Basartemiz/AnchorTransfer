#!/usr/bin/env bash
# Setup environment for DrugBAN paper replication.
# Installs all required packages into the current Python environment.
set -euo pipefail

echo "=== DrugBAN Paper Replication Setup ==="

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 10), f'Python 3.10+ required, got {sys.version}'"
echo "Python: $(python3 --version)"

# Install dependencies
pip install torch torchvision torchaudio --quiet 2>/dev/null || true
pip install torch_geometric rdkit scikit-learn pandas numpy tqdm --quiet

# Verify
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
from torch_geometric.data import Data
from rdkit import Chem
from sklearn.metrics import roc_auc_score
import tqdm
print('All dependencies OK')
"

echo "=== Setup complete ==="
