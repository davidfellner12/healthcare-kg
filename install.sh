#!/bin/bash
set -e

echo "============================================"
echo " Healthcare KG - Linux/Mac Setup"
echo "============================================"

echo ""
echo "[1/5] Core packages..."
pip install rdflib requests pandas shapely pyproj flask flask-cors \
    folium matplotlib seaborn ipykernel jupyter tqdm python-dotenv \
    pyyaml scikit-learn numpy

echo ""
echo "[2/5] PyTorch (CPU)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "[3/5] PyKEEN (TransE)..."
pip install pykeen

echo ""
echo "[4/5] PyTorch Geometric (GraphSAGE)..."
pip install torch_geometric
pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html || \
    echo "WARNING: Optional PyG extensions failed - core GraphSAGE still works"

echo ""
echo "[5/5] geopandas..."
pip install geopandas || echo "WARNING: geopandas optional, skipping"

echo ""
echo "============================================"
echo " Done! Run:"
echo "   python scripts/build_kg.py --skip-ml"
echo "   python src/api/app.py"
echo "============================================"
