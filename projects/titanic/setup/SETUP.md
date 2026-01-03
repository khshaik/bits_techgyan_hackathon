# Titanic Project - Setup Guide

This guide provides step-by-step instructions to set up and run the Titanic Survival Prediction project using Jupyter Notebook or Python script.

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.7 or higher**
  ```bash
  python3 --version
  ```
- **pip** (Python package manager)
  ```bash
  pip --version
  ```
- **Internet connection** (to download packages and data)
- **Text editor or IDE** (optional, for viewing/editing code)

---

## 🔧 Step 1: Navigate to Project Directory

```bash
# From BITS_Hackathon root directory
cd titanic

# Verify you're in correct directory
pwd  # macOS/Linux
cd   # Windows
```

---

## 🌐 Step 2: Create Virtual Environment

Virtual environments isolate project dependencies and prevent conflicts.

```bash
# Create virtual environment
python3 -m venv venv

# Verify creation
ls -la venv  # macOS/Linux
dir venv     # Windows
```

---

## ✅ Step 3: Activate Virtual Environment

### macOS/Linux
```bash
source venv/bin/activate

# Verify activation (should see (venv) in terminal prompt)
which python
```

### Windows (Command Prompt)
```cmd
venv\Scripts\activate

# Verify activation (should see (venv) in terminal prompt)
where python
```

### Windows (PowerShell)
```powershell
venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📦 Step 4: Install Dependencies

### Option A: Install from requirements file (Recommended)

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

### Option B: Manual installation

```bash
# Core data science libraries
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0

# Visualization
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Jupyter for notebook
pip install jupyter==1.0.0
pip install notebook==7.0.0
```

### Option C: Install with specific versions

```bash
pip install -r requirements.txt --no-cache-dir
```

---

## ✨ Step 5: Verify Installation

### Check installed packages

```bash
# List all installed packages
pip list

# Check specific packages
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import jupyter; print(f'Jupyter: {jupyter.__version__}')"
```

### Test imports

Create a test file `test_setup.py`:

```python
#!/usr/bin/env python3
"""Test if all required packages are installed."""

import sys

packages = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical computing',
    'sklearn': 'Machine learning',
    'matplotlib': 'Visualization',
    'seaborn': 'Statistical visualization',
    'jupyter': 'Notebook environment'
}

print("=" * 60)
print("TESTING TITANIC PROJECT SETUP")
print("=" * 60)

failed = []
for package, description in packages.items():
    try:
        __import__(package)
        print(f"✓ {package:15} - {description}")
    except ImportError:
        print(f"✗ {package:15} - {description} [NOT INSTALLED]")
        failed.append(package)

print("=" * 60)

if failed:
    print(f"\n⚠️  Missing packages: {', '.join(failed)}")
    print(f"Install with: pip install {' '.join(failed)}")
    sys.exit(1)
else:
    print("\n✓ All packages installed successfully!")
    print("Ready to run the Titanic project!")
    sys.exit(0)
```

Run the test:

```bash
python test_setup.py
```

---

## 🚀 Step 6: Run the Project

### Option A: Run as Jupyter Notebook (Recommended for Learning)

```bash
# Start Jupyter Notebook server
jupyter notebook

# Expected output:
# [I 12:34:56.789 NotebookApp] Serving notebooks from local directory: /path/to/titanic
# [I 12:34:56.789 NotebookApp] Jupyter Notebook 7.0.0 is running at:
# [I 12:34:56.789 NotebookApp] http://localhost:8888/?token=...
```

**Access the notebook**:
1. Browser opens automatically at `http://localhost:8888`
2. If not, manually navigate to the URL shown in terminal
3. Click on `BITS_AIML_Titanic_Jan3rd2026.ipynb`
4. Execute cells sequentially using Shift+Enter

**Notebook Features**:
- Interactive cell execution
- Inline visualizations
- Markdown documentation
- Easy experimentation
- Code modification and testing

### Option B: Run as Python Script (For Batch Processing)

```bash
# Run Python script directly
python BITS_AIML_Titanic_Jan3rd2026.py

# Expected output:
# ================================================================================
# TITANIC DATASET - FIRST 5 ROWS
# ================================================================================
# [Dataset preview]
# ...
# [Analysis results]
# [Visualizations open in separate windows]
```

**Script Features**:
- Complete pipeline execution
- Console output of all results
- Visualizations in separate windows
- No notebook interface required
- Batch processing capability

---

## 🎯 Step 7: Using the Project

### Jupyter Notebook Workflow

1. **Open Notebook**
   - Click on `BITS_AIML_Titanic_Jan3rd2026.ipynb`
   - Notebook loads in browser

2. **Execute Cells**
   - Click on cell or press Ctrl+A to select all
   - Press Shift+Enter to execute
   - Or use Cell menu → Run All

3. **View Results**
   - Output appears below each cell
   - Visualizations display inline
   - Modify code and re-run

4. **Experiment**
   - Change parameters and observe results
   - Add new cells for custom analysis
   - Save modified notebook

### Python Script Workflow

1. **Run Script**
   ```bash
   python BITS_AIML_Titanic_Jan3rd2026.py
   ```

2. **View Console Output**
   - Dataset information
   - Statistical summaries
   - Model metrics
   - Analysis results

3. **View Visualizations**
   - Matplotlib windows open automatically
   - Close windows to continue execution
   - All plots display during execution

---

## 🔧 Step 8: Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'jupyter'"

**Solution**:
```bash
# Verify virtual environment is activated
# You should see (venv) in terminal prompt

# Install Jupyter
pip install jupyter==1.0.0
pip install notebook==7.0.0

# Or reinstall all dependencies
pip install -r requirements.txt
```

### Issue: "No module named 'sklearn'"

**Solution**:
```bash
# Install scikit-learn
pip install scikit-learn==1.3.0

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

### Issue: "Jupyter notebook command not found"

**Solution**:
```bash
# Use Python module syntax
python -m jupyter notebook

# Or ensure Jupyter is installed
pip install --upgrade jupyter notebook
```

### Issue: "Data loading fails" (GitHub connection error)

**Solution**:
```bash
# Check internet connection
ping github.com

# Verify URL is accessible
# Open in browser: https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv

# Check pandas version
pip install --upgrade pandas
```

### Issue: "Matplotlib display error"

**Solution - For Jupyter**:
```python
# Add this at the beginning of notebook
%matplotlib inline
```

**Solution - For Python Script**:
```bash
# Ensure matplotlib is installed
pip install matplotlib==3.7.2

# On macOS, may need to use different backend
# Add to script: import matplotlib; matplotlib.use('TkAgg')
```

### Issue: "Port 8888 already in use" (Jupyter)

**Solution 1: Use different port**
```bash
jupyter notebook --port 8889
```

**Solution 2: Find and kill the process**
```bash
# macOS/Linux
lsof -i :8888
kill -9 <PID>

# Windows
netstat -ano | findstr :8888
taskkill /PID <PID> /F
```

### Issue: "Kernel dead" or "Kernel restarting" (Jupyter)

**Solution**:
```bash
# Restart kernel
# Kernel menu → Restart

# Or restart Jupyter
# Stop Jupyter (Ctrl+C)
# Start again: jupyter notebook
```

### Issue: "MemoryError" with large visualizations

**Solution**:
```bash
# Close unnecessary applications
# Reduce figure size in code:
plt.rcParams['figure.figsize'] = (8, 4)  # Instead of (12, 6)
```

---

## 📊 Step 9: Verify Full Setup

Run this checklist:

- [ ] Python 3.7+ installed
- [ ] Virtual environment created
- [ ] Virtual environment activated (see (venv) in prompt)
- [ ] pip upgraded
- [ ] All dependencies installed
- [ ] test_setup.py passes
- [ ] Jupyter starts without errors (for notebook)
- [ ] Python script runs without errors (for script)
- [ ] Can access http://localhost:8888 (for notebook)
- [ ] All workflow steps execute successfully
- [ ] Visualizations display correctly
- [ ] No data loading errors

---

## 🔄 Step 10: Alternative Setup Methods

### Using Anaconda

```bash
# Create conda environment
conda create -n titanic python=3.9

# Activate environment
conda activate titanic

# Install dependencies
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn jupyter

# Run Jupyter
jupyter notebook

# Or run Python script
python BITS_AIML_Titanic_Jan3rd2026.py
```

### Using Docker

```bash
# Create Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]

# Build image
docker build -t titanic-project .

# Run container
docker run -p 8888:8888 titanic-project
```

### Using Google Colab (Cloud-based)

```bash
# 1. Go to https://colab.research.google.com/
# 2. Create new notebook
# 3. Upload notebook or paste code
# 4. Run cells in cloud (no local setup needed)
# 5. All libraries pre-installed
```

---

## 💾 Step 11: Managing the Environment

### Save environment configuration

```bash
# Export current environment
pip freeze > requirements_current.txt

# Later, restore from file
pip install -r requirements_current.txt
```

### Update packages

```bash
# Check for updates
pip list --outdated

# Update specific package
pip install --upgrade pandas

# Update all packages
pip install --upgrade -r requirements.txt
```

### Delete environment

```bash
# Deactivate first
deactivate

# Remove environment directory
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows
```

---

## 🎓 Step 12: Next Steps

After successful setup:

1. **Explore the Code**
   - Read `BITS_AIML_Titanic_Jan3rd2026.ipynb` or `.py`
   - Understand the data pipeline
   - Study the decision tree implementation
   - Review comments and documentation

2. **Experiment**
   - Modify hyperparameters (max_depth, min_samples_split)
   - Try different features
   - Change train/test split ratio
   - Implement cross-validation

3. **Extend the Project**
   - Implement ensemble methods (Random Forest)
   - Add hyperparameter tuning (GridSearchCV)
   - Create cross-validation
   - Build web API for predictions

4. **Learn More**
   - Read the main README.md
   - Study ARCHITECTURE.md
   - Review code comments
   - Explore scikit-learn documentation

---

## 📚 Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook

# Run Python script
python BITS_AIML_Titanic_Jan3rd2026.py

# Test setup
python test_setup.py

# Deactivate environment
deactivate

# Check Python version
python --version

# List installed packages
pip list

# Upgrade pip
pip install --upgrade pip
```

---

## 🆘 Getting Help

If you encounter issues:

1. **Check error messages carefully** - They often indicate the solution
2. **Verify prerequisites** - Ensure Python 3.7+ and pip are installed
3. **Check virtual environment** - Confirm it's activated (see (venv) in prompt)
4. **Review documentation** - Check README.md and ARCHITECTURE.md
5. **Test imports** - Run test_setup.py to verify packages
6. **Check internet connection** - Required for data loading
7. **Review scikit-learn docs** - https://scikit-learn.org/
8. **Check Jupyter docs** - https://jupyter.org/

---

## ✅ Setup Completion

Once you complete all steps:

1. Virtual environment is active
2. All dependencies are installed
3. Jupyter or Python script runs without errors
4. Data loads successfully from GitHub
5. All workflow steps execute
6. Visualizations display correctly
7. Model training completes
8. Predictions work correctly

You're ready to explore and learn from the Titanic Survival Prediction project!

---

## 📝 Common Jupyter Shortcuts

| Shortcut | Action |
|----------|--------|
| Shift+Enter | Execute cell and move to next |
| Ctrl+Enter | Execute cell and stay |
| Alt+Enter | Execute cell and insert new below |
| Ctrl+S | Save notebook |
| A | Insert cell above |
| B | Insert cell below |
| D,D | Delete cell |
| M | Change to Markdown |
| Y | Change to Code |
| Ctrl+/ | Comment/uncomment |

---

**Last Updated**: January 2026
**Version**: 1.0
**Status**: Ready for Use
