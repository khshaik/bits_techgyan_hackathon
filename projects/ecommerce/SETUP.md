# E-commerce Project - Setup Guide

This guide provides step-by-step instructions to set up and run the E-commerce Customer Analysis Streamlit dashboard.

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
- **Available port 8501** (for Streamlit development server)

---

## 🔧 Step 1: Navigate to Project Directory

```bash
# From BITS_Hackathon root directory
cd ecommerce

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
pip install -r requirements_streamlit.txt

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

# Streamlit framework
pip install streamlit==1.28.1
```

### Option C: Install with specific versions

```bash
pip install -r requirements_streamlit.txt --no-cache-dir
```

---

## ✨ Step 5: Verify Installation

### Check installed packages

```bash
# List all installed packages
pip list

# Check specific packages
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
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
    'streamlit': 'Dashboard framework'
}

print("=" * 60)
print("TESTING E-COMMERCE PROJECT SETUP")
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
    print("Ready to run the Streamlit dashboard!")
    sys.exit(0)
```

Run the test:

```bash
python test_setup.py
```

---

## 🚀 Step 6: Run Streamlit Application

### Start the development server

```bash
# Run Streamlit application
streamlit run ecommerce_customer_streamlit_analysis.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://XXX.XXX.XXX.XXX:8501
```

### Access the application

1. Browser opens automatically at `http://localhost:8501`
2. If not, manually navigate to: `http://localhost:8501`
3. You should see the E-commerce Analysis dashboard

---

## 🎯 Step 7: Using the Dashboard

### Navigation

1. **Sidebar Menu**: Select different analysis steps
2. **Main Area**: View results and interact with controls
3. **Widgets**: Use sliders and inputs for predictions

### Workflow Steps

1. **Analyze**: View dataset overview and statistics
2. **Clean**: See data cleaning results
3. **Visualize**: Explore data distributions and correlations
4. **Train**: Check model coefficients and parameters
5. **Test**: View performance metrics and visualizations
6. **Deploy**: Make predictions with interactive inputs

### Making Predictions

1. Navigate to "Deploy" step
2. Use sidebar sliders to adjust customer characteristics
3. View real-time spending predictions
4. Experiment with different customer profiles

---

## 🔧 Step 8: Troubleshooting

### Issue: "Port 8501 already in use"

**Solution 1: Use different port**
```bash
streamlit run ecommerce_customer_streamlit_analysis.py --server.port 8502
```

**Solution 2: Find and kill the process**
```bash
# macOS/Linux
lsof -i :8501
kill -9 <PID>

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```bash
# Verify virtual environment is activated
# You should see (venv) in terminal prompt

# Reinstall dependencies
pip install -r requirements_streamlit.txt

# Or install Streamlit directly
pip install streamlit==1.28.1
```

### Issue: "No module named 'sklearn'"

**Solution**:
```bash
# Install scikit-learn
pip install scikit-learn==1.3.0

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

### Issue: "Connection refused" when accessing localhost:8501

**Solution**:
```bash
# Ensure Streamlit app is running
# Check terminal for "You can now view your Streamlit app"

# If not running, start it:
streamlit run ecommerce_customer_streamlit_analysis.py

# Try different URL:
# http://127.0.0.1:8501 (instead of localhost)
```

### Issue: "Data loading fails"

**Solution**:
```bash
# Check internet connection
ping github.com

# Verify GitHub URL is accessible
# Check pandas version
pip install --upgrade pandas
```

### Issue: "Streamlit app keeps rerunning"

**Solution**:
```bash
# This is normal Streamlit behavior
# The app reruns when you interact with widgets
# To optimize, use @st.cache_data decorator for expensive operations

# Example:
@st.cache_data
def load_data():
    return pd.read_csv(url)
```

### Issue: "Widget value not updating"

**Solution**:
```bash
# Ensure you're using st.session_state for persistent values
# Or use callback functions with widgets

# Example:
value = st.slider("Select value", 0, 100, key="my_slider")
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
- [ ] Streamlit app starts without errors
- [ ] Can access http://localhost:8501
- [ ] All workflow steps execute successfully
- [ ] Predictions work correctly
- [ ] Dashboard is interactive

---

## 🔄 Step 10: Alternative Setup Methods

### Using Anaconda

```bash
# Create conda environment
conda create -n ecommerce python=3.9

# Activate environment
conda activate ecommerce

# Install dependencies
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn streamlit

# Run application
streamlit run ecommerce_customer_streamlit_analysis.py
```

### Using Docker

```bash
# Create Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "ecommerce_customer_streamlit_analysis.py"]

# Build image
docker build -t ecommerce-dashboard .

# Run container
docker run -p 8501:8501 ecommerce-dashboard
```

### Deploy to Streamlit Cloud

```bash
# 1. Push code to GitHub repository
git push origin main

# 2. Go to https://share.streamlit.io/
# 3. Click "New app"
# 4. Select your GitHub repository
# 5. Select main branch and app file
# 6. Click "Deploy"
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
pip install --upgrade streamlit

# Update all packages
pip install --upgrade -r requirements_streamlit.txt
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

1. **Explore the code**
   - Read `ecommerce_customer_streamlit_analysis.py`
   - Understand the data pipeline
   - Study the Streamlit components

2. **Experiment**
   - Modify dashboard layout
   - Change visualization styles
   - Adjust model parameters
   - Add custom widgets

3. **Extend the project**
   - Add customer segmentation
   - Implement clustering analysis
   - Create export functionality
   - Add historical tracking

4. **Learn more**
   - Read the main README.md
   - Study ARCHITECTURE.md
   - Review code comments
   - Explore Streamlit documentation

---

## 📚 Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements_streamlit.txt

# Run application
streamlit run ecommerce_customer_streamlit_analysis.py

# Run on different port
streamlit run app.py --server.port 8502

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
3. **Check virtual environment** - Confirm it's activated
4. **Review documentation** - Check README.md and ARCHITECTURE.md
5. **Test imports** - Run test_setup.py to verify packages
6. **Check internet connection** - Required for data loading
7. **Review Streamlit docs** - https://docs.streamlit.io/

---

## ✅ Setup Completion

Once you complete all steps:

1. Virtual environment is active
2. All dependencies are installed
3. Streamlit app runs without errors
4. Dashboard is accessible
5. All workflow steps execute
6. Predictions work correctly
7. Interactive widgets respond properly

You're ready to explore and learn from the E-commerce Customer Analysis project!

---

**Last Updated**: January 2026
**Version**: 1.0
**Status**: Ready for Use
