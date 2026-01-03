# Advertising Project - Setup Guide

This guide provides step-by-step instructions to set up and run the Advertising Analysis Flask application.

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
- **Available port 5000** (for Flask development server)

---

## 🔧 Step 1: Navigate to Project Directory

```bash
# From BITS_Hackathon root directory
cd advertising

# Verify you're in correct directory
pwd  # macOS/Linux
cd   # Windows
```

---

## 🌐 Step 2: Create Virtual Environment

Virtual environments isolate project dependencies and prevent conflicts with other projects.

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
pip install -r requirements_flask.txt

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

# Web framework
pip install flask==2.3.2

# Utilities
pip install werkzeug==2.3.6
```

### Option C: Install with specific versions

```bash
pip install -r requirements_flask.txt --no-cache-dir
```

---

## ✨ Step 5: Verify Installation

### Check installed packages

```bash
# List all installed packages
pip list

# Check specific packages
python -c "import flask; print(f'Flask: {flask.__version__}')"
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
    'flask': 'Web framework'
}

print("=" * 60)
print("TESTING ADVERTISING PROJECT SETUP")
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
    print("Ready to run the Flask application!")
    sys.exit(0)
```

Run the test:

```bash
python test_setup.py
```

---

## 🚀 Step 6: Run Flask Application

### Start the development server

```bash
# Run Flask application
python newspaper_advertising_flask_analysis.py

# Expected output:
# * Serving Flask app 'newspaper_advertising_flask_analysis'
# * Debug mode: on
# * Running on http://127.0.0.1:5000
# Press CTRL+C to quit
```

### Access the application

1. Open your web browser
2. Navigate to: `http://localhost:5000`
3. You should see the Advertising Analysis home page

---

## 🎯 Step 7: Using the Application

### Workflow Steps

1. **Home Page**
   - Overview of the project
   - Navigation to different steps

2. **Step 1: Analyze**
   - Click "Analyze Dataset" button
   - View dataset statistics and information
   - Check for missing values

3. **Step 2: Clean**
   - Click "Clean Data" button
   - See data cleaning results
   - Verify data quality

4. **Step 3: Visualize**
   - Click "Visualize Data" button
   - View distribution plots
   - See correlation heatmap
   - Analyze feature relationships

5. **Step 4: Train**
   - Click "Train Model" button
   - See model coefficients
   - Check training parameters

6. **Step 5: Test**
   - Click "Test Model" button
   - View performance metrics
   - See actual vs predicted plots
   - Analyze residuals

7. **Step 6: Deploy**
   - Click "Make Predictions" button
   - Enter advertising spend values
   - Get sales predictions

---

## 🔧 Step 8: Troubleshooting

### Issue: "Port 5000 already in use"

**Solution 1: Find and kill the process**
```bash
# macOS/Linux
lsof -i :5000
kill -9 <PID>

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Solution 2: Use different port**
```bash
# Edit newspaper_advertising_flask_analysis.py
# Change: app.run(debug=True, port=5000)
# To: app.run(debug=True, port=5001)
```

### Issue: "ModuleNotFoundError: No module named 'flask'"

**Solution**:
```bash
# Verify virtual environment is activated
# You should see (venv) in terminal prompt

# Reinstall dependencies
pip install -r requirements_flask.txt

# Or install Flask directly
pip install flask==2.3.2
```

### Issue: "No module named 'sklearn'"

**Solution**:
```bash
# Install scikit-learn
pip install scikit-learn==1.3.0

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

### Issue: "Connection refused" when accessing localhost:5000

**Solution**:
```bash
# Ensure Flask app is running
# Check terminal for "Running on http://127.0.0.1:5000"

# If not running, start it:
python newspaper_advertising_flask_analysis.py

# Try different URL:
# http://127.0.0.1:5000 (instead of localhost)
```

### Issue: "Data loading fails" (GitHub connection error)

**Solution**:
```bash
# Check internet connection
ping github.com

# Verify URL is accessible
# Open in browser: https://raw.githubusercontent.com/erkansirin78/datasets/master/Advertising.csv

# Check pandas version
pip install --upgrade pandas
```

### Issue: "Matplotlib backend error"

**Solution**:
```bash
# The Flask app uses 'Agg' backend (non-interactive)
# This is already configured in the code

# If you still get errors, ensure matplotlib is installed:
pip install matplotlib==3.7.2
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
- [ ] Flask app starts without errors
- [ ] Can access http://localhost:5000
- [ ] All workflow steps execute successfully
- [ ] Predictions work correctly

---

## 🔄 Step 10: Alternative Setup Methods

### Using Anaconda

```bash
# Create conda environment
conda create -n advertising python=3.9

# Activate environment
conda activate advertising

# Install dependencies
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn flask

# Run application
python newspaper_advertising_flask_analysis.py
```

### Using Docker

```bash
# Create Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_flask.txt .
RUN pip install -r requirements_flask.txt
COPY . .
CMD ["python", "newspaper_advertising_flask_analysis.py"]

# Build image
docker build -t advertising-app .

# Run container
docker run -p 5000:5000 advertising-app
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
pip install --upgrade flask

# Update all packages
pip install --upgrade -r requirements_flask.txt
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
   - Read `newspaper_advertising_flask_analysis.py`
   - Understand the data pipeline
   - Study the Flask routes

2. **Experiment**
   - Modify HTML template
   - Change visualization styles
   - Adjust model parameters

3. **Extend the project**
   - Add more visualizations
   - Implement additional metrics
   - Create prediction history
   - Add database integration

4. **Learn more**
   - Read the main README.md
   - Study ARCHITECTURE.md
   - Review code comments

---

## 📚 Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements_flask.txt

# Run application
python newspaper_advertising_flask_analysis.py

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

---

## ✅ Setup Completion

Once you complete all steps:

1. Virtual environment is active
2. All dependencies are installed
3. Flask app runs without errors
4. Web interface is accessible
5. All workflow steps execute
6. Predictions work correctly

You're ready to explore and learn from the Advertising Analysis project!

---

**Last Updated**: January 2026
**Version**: 1.0
**Status**: Ready for Use
