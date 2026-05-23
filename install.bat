@echo off
echo ============================================
echo  Healthcare KG - Windows Setup Script
echo ============================================
echo.

REM Step 1: Core packages (no special deps)
echo [1/5] Installing core packages...
pip install rdflib pyparsing==3.1.1 requests pandas shapely pyproj flask flask-cors folium matplotlib seaborn ipykernel jupyter tqdm python-dotenv pyyaml scikit-learn numpy
if %errorlevel% neq 0 (echo ERROR in step 1 & pause & exit /b 1)

REM Step 2: PyTorch CPU (must come before torch-geometric, torch-scatter etc.)
echo.
echo [2/5] Installing PyTorch (CPU only - smaller, no CUDA needed)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (echo ERROR in step 2 & pause & exit /b 1)

REM Step 3: PyKEEN for TransE
echo.
echo [3/5] Installing PyKEEN (TransE embeddings)...
pip install pykeen
if %errorlevel% neq 0 (echo ERROR in step 3 & pause & exit /b 1)

REM Step 4: PyTorch Geometric (needs torch already installed)
echo.
echo [4/5] Installing PyTorch Geometric (GraphSAGE)...
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
if %errorlevel% neq 0 (
    echo WARNING: Optional PyG extensions failed - basic GraphSAGE will still work
)

REM Step 5: geopandas last (sometimes tricky on Windows)
echo.
echo [5/5] Installing geopandas...
pip install geopandas
if %errorlevel% neq 0 (
    echo WARNING: geopandas failed - not required for core functionality
)

echo.
echo ============================================
echo  Installation complete!
echo.
echo  Next steps:
echo    python scripts/build_kg.py --skip-ml
echo    python src/api/app.py
echo    Open http://localhost:5000
echo ============================================
pause
