#!/usr/bin/env bash
# Instala el entorno del EHO en el servidor (Ubuntu, 4 × RTX A4000) y lo verifica.
# Uso:  bash scripts/instalar_servidor.sh        (desde la raíz del repositorio)
set -euo pipefail

ENV=hemotwin
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV} "; then
  conda create -y -n "$ENV" python=3.12
fi
conda activate "$ENV"

# PyTorch de PyPI para Linux x86_64 trae CUDA incluido; no hace falta el CUDA del sistema.
pip install --upgrade pip
pip install torch numpy scipy pytest duckdb pandas pyarrow matplotlib

echo "=== PyTorch y GPU ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA disponible:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
PY

echo "=== MPI (para HemoCell/LAMMPS) ==="
which mpirun mpicc 2>/dev/null || echo "MPI no instalado (se puede instalar con: conda install -c conda-forge openmpi)"

echo "=== Pruebas (CPU y GPU) ==="
EHO_DEVICE=cuda:0 python -m pytest -q tests

echo "=== Comparación de velocidad CPU frente a GPU ==="
python scripts/diagnostico_velocidad.py

echo "Listo. Active el entorno con: conda activate $ENV"
