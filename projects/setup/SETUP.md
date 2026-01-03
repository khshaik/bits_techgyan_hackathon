# General Setup Guide - BITS Hackathon Projects

This guide covers the general setup process applicable to all sub-projects. For project-specific setup, refer to individual project SETUP.md files.

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.7 or higher** installed
  - Check version: `python3 --version`
- **pip** (Python package manager)
  - Check version: `pip --version`
- **Git** (optional, for version control)
- **Text editor or IDE** (VS Code, PyCharm, etc.)
- **Internet connection** (for downloading datasets and packages)

---

## 🔧 Step 1: Environment Setup

### 1.1 Create a Virtual Environment

Virtual environments isolate project dependencies and prevent conflicts.

```bash
# Navigate to project directory
cd BITS_Hackathon

# Create virtual environment (replace 'myenv' with your preferred name)
python3 -m venv myenv

# Activate virtual environment
# On macOS/Linux:
source myenv/bin/activate

# On Windows:
myenv\Scripts\activate

# Verify activation (you should see (myenv) in your terminal)
which python
```

### 1.2 Upgrade pip

Ensure you have the latest pip version:

```bash
pip install --upgrade pip
```

---

## 📦 Step 2: Install Dependencies

### Option A: Install All Projects (Recommended for Learning)

```bash
# Install all requirements
pip install pandas numpy scikit-learn matplotlib seaborn jupyter flask streamlit

# Or use individual requirements files
pip install -r advertising/requirements_flask.txt
pip install -r ecommerce/requirements_streamlit.txt
pip install -r titanic/requirements.txt
```

### Option B: Install Project-Specific Dependencies

Install only what you need for your chosen project:

```bash
# For Advertising (Flask)
pip install -r advertising/requirements_flask.txt

# For E-commerce (Streamlit)
pip install -r ecommerce/requirements_streamlit.txt

# For Titanic (Jupyter/Python)
pip install -r titanic/requirements.txt
```

### Option C: Manual Installation

```bash
# Core data science libraries
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0

# Visualization
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Jupyter (for notebooks)
pip install jupyter==1.0.0
pip install notebook==7.0.0

# Flask (for web app)
pip install flask==2.3.2

# Streamlit (for dashboard)
pip install streamlit==1.28.1
```

---

## ✅ Step 3: Verify Installation

### 3.1 Check Python Packages

```bash
# List installed packages
pip list

# Check specific packages
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
```

### 3.2 Test Import Statements

Create a test file `test_imports.py`:

```python
#!/usr/bin/env python3
"""Test if all required packages are installed correctly."""

import sys

packages = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical computing',
    'sklearn': 'Machine learning',
    'matplotlib': 'Visualization',
    'seaborn': 'Statistical visualization',
    'flask': 'Web framework (Advertising)',
    'streamlit': 'Dashboard framework (E-commerce)',
    'jupyter': 'Notebook environment'
}

print("=" * 60)
print("TESTING PACKAGE IMPORTS")
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
    sys.exit(0)
```

Run the test:

```bash
python test_imports.py
```

---

## 🚀 Step 4: Running Projects

### 4.1 Jupyter Notebook (Titanic Project)

```bash
# Navigate to project directory
cd titanic

# Start Jupyter Notebook server
jupyter notebook

# In browser, open: http://localhost:8888
# Click on BITS_AIML_Titanic_Jan3rd2026.ipynb
```

**Alternative - Run as Python Script**:

```bash
python BITS_AIML_Titanic_Jan3rd2026.py
```

### 4.2 Flask Web Application (Advertising Project)

```bash
# Navigate to project directory
cd advertising

# Run Flask app
python newspaper_advertising_flask_analysis.py

# Open browser: http://localhost:5000
# Press Ctrl+C to stop the server
```

### 4.3 Streamlit Dashboard (E-commerce Project)

```bash
# Navigate to project directory
cd ecommerce

# Run Streamlit app
streamlit run ecommerce_customer_streamlit_analysis.py

# Opens automatically at http://localhost:8501
# Press Ctrl+C to stop the server
```

---

## 🔄 Step 5: Deactivate Virtual Environment

When finished working:

```bash
# Deactivate virtual environment
deactivate

# Verify deactivation (should not see (myenv) in terminal)
```

---

## 📝 Step 6: Project-Specific Setup

After completing general setup, follow project-specific instructions:

- **Advertising**: See `advertising/SETUP.md`
- **E-commerce**: See `ecommerce/SETUP.md`
- **Titanic**: See `titanic/SETUP.md`

---

## 🐛 Troubleshooting

### Issue: "Command not found: python3"

**Solution**:
```bash
# Check if Python is installed
python --version

# Use 'python' instead of 'python3'
python -m venv myenv
```

### Issue: "pip: command not found"

**Solution**:
```bash
# Use Python module syntax
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn
```

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution**:
```bash
# Ensure virtual environment is activated
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate     # Windows

# Reinstall packages
pip install -r requirements.txt
```

### Issue: "Port already in use" (Flask/Streamlit)

**Solution - Flask**:
```bash
# Change port in code or use:
python -c "from newspaper_advertising_flask_analysis import app; app.run(port=5001)"

# Or find and kill process:
lsof -i :5000
kill -9 <PID>
```

**Solution - Streamlit**:
```bash
streamlit run app.py --server.port 8502
```

### Issue: "Jupyter notebook command not found"

**Solution**:
```bash
# Install Jupyter
pip install jupyter notebook

# Run with Python module
python -m jupyter notebook
```

### Issue: "SSL: CERTIFICATE_VERIFY_FAILED" (macOS)

**Solution**:
```bash
# Install certificates
/Applications/Python\ 3.x/Install\ Certificates.command

# Or disable SSL verification (not recommended for production)
pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pandas
```

---

## 📊 Verifying Setup Completion

Run this checklist to verify everything is set up correctly:

- [ ] Python 3.7+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list` shows packages)
- [ ] Test imports pass (`python test_imports.py`)
- [ ] Can start Jupyter Notebook
- [ ] Can run Flask app (port 5000 accessible)
- [ ] Can run Streamlit app (port 8501 accessible)
- [ ] Can access http://localhost:8888 (Jupyter)
- [ ] Can access http://localhost:5000 (Flask)
- [ ] Can access http://localhost:8501 (Streamlit)

---

## 💾 Managing Virtual Environments

### Save Current Environment

```bash
# Export current environment
pip freeze > requirements_custom.txt

# Later, restore from file
pip install -r requirements_custom.txt
```

### Delete Virtual Environment

```bash
# Simply remove the directory
rm -rf myenv  # macOS/Linux
rmdir myenv   # Windows
```

### Create Multiple Environments

```bash
# For different projects
python3 -m venv advertising_env
python3 -m venv ecommerce_env
python3 -m venv titanic_env

# Activate as needed
source advertising_env/bin/activate
```

---

## 🔐 Best Practices

1. **Always use virtual environments** - Prevents dependency conflicts
2. **Keep requirements.txt updated** - Document all dependencies
3. **Use specific versions** - Ensures reproducibility
4. **Document setup steps** - Help others reproduce your environment
5. **Test after installation** - Verify everything works
6. **Keep dependencies minimal** - Only install what you need

---

## 📚 Additional Resources

### Python Virtual Environments
- [Official venv documentation](https://docs.python.org/3/library/venv.html)
- [Virtual Environment Guide](https://realpython.com/python-virtual-environments-a-primer/)

### Package Management
- [pip documentation](https://pip.pypa.io/)
- [PyPI - Python Package Index](https://pypi.org/)

### Development Tools
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## ✨ Quick Reference

```bash
# Complete setup in 5 commands
python3 -m venv myenv
source myenv/bin/activate  # or myenv\Scripts\activate on Windows
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter flask streamlit
python test_imports.py
```

---

**Last Updated**: January 2026
**Version**: 1.0
**Status**: Ready for Use
