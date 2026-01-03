# Titanic Survival Prediction - Jupyter Notebook & Python Script Deployment Guide

## 📊 Project Overview

This project performs comprehensive analysis on the Titanic dataset using Decision Tree Classification to predict passenger survival outcomes based on demographic and ticket information.

**Live Dataset Source:** https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv

---

## 🎯 Workflow Steps

### 1. **ANALYZE** - Dataset Exploration
- Load Titanic dataset from GitHub
- Display dataset dimensions and structure
- Show column information and data types
- Generate statistical summaries
- Identify missing values and data quality issues
- Analyze survival distribution

### 2. **CLEAN** - Data Preprocessing
- Handle missing Age values (fill with class-specific mean)
- Handle missing Embarked values (fill with mode)
- Remove rows with remaining missing values
- Remove duplicate records
- Encode categorical variables (Sex, Embarked)
- Validate data integrity

### 3. **VISUALIZE** - Data Exploration
- Distribution of survival outcomes
- Survival by passenger class
- Survival by gender
- Age distribution and relationship with survival
- Family size impact on survival
- Fare distribution analysis

### 4. **TRAIN** - Model Development
- Prepare features and target variable
- Split data: 67% training, 33% testing (random_state=3)
- Encode categorical variables using LabelEncoder
- Train Decision Tree Classifier
- Display model coefficients and feature importance

### 5. **TEST** - Model Evaluation
- Calculate comprehensive performance metrics:
  - **Accuracy Score**
  - **Precision Score**
  - **Recall Score**
  - **F1-Score**
- Generate confusion matrix
- Visualize actual vs predicted values
- Analyze model performance
- Compare training vs testing performance

### 6. **DEPLOY** - Make Predictions
- Interactive prediction interface
- Input passenger characteristics
- Real-time survival prediction
- Display prediction results with confidence

---

## 📋 Performance Metrics

The application calculates and displays:

### Classification Metrics
- **Accuracy:** Proportion of correct predictions (0-1 scale)
- **Precision:** Of predicted survivors, how many actually survived
- **Recall:** Of actual survivors, how many were correctly predicted
- **F1-Score:** Harmonic mean of precision and recall

### Visualizations
- Confusion matrix heatmap
- Actual vs Predicted scatter plots
- Feature importance bar charts
- Survival distribution plots
- Class-based survival analysis

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Git (optional, for cloning repository)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install jupyter==1.0.0
pip install notebook==7.0.0
```

### Step 2: Project Structure

Ensure your project structure looks like this:

```
BITS_Hackathon/titanic/
├── BITS_AIML_Titanic_Jan3rd2026.ipynb      # Jupyter notebook
├── BITS_AIML_Titanic_Jan3rd2026.py         # Python script
├── requirements.txt                         # Python dependencies
└── TITANIC_JUPYTER_DEPLOYMENT.md            # This file
```

### Step 3: Run the Application

**Option A: Jupyter Notebook**

```bash
jupyter notebook BITS_AIML_Titanic_Jan3rd2026.ipynb
```

The notebook will open in your browser at `http://localhost:8888`

**Option B: Python Script**

```bash
python BITS_AIML_Titanic_Jan3rd2026.py
```

---

## 🚀 How to Use the Application

### Using Jupyter Notebook

#### 1. Access the Notebook
- Open your browser and navigate to `http://localhost:8888`
- Click on `BITS_AIML_Titanic_Jan3rd2026.ipynb`
- Notebook loads in the browser

#### 2. Execute Cells Sequentially

Each cell represents a workflow step:

##### Step 1: ANALYZE
- Loads dataset from GitHub
- Displays dataset dimensions (891 rows, 12 columns)
- Shows column information and statistics
- Identifies missing values (Age: 177, Cabin: 687, Embarked: 2)

##### Step 2: CLEAN
- Fills missing Age values with class-specific mean
- Fills missing Embarked values with mode ('S')
- Removes rows with remaining missing values
- Removes duplicate records
- Displays cleaning statistics

##### Step 3: VISUALIZE
- Survival distribution (pie chart and counts)
- Survival by passenger class (bar chart)
- Survival by gender (bar chart)
- Age distribution and survival relationship
- Family size impact visualization

##### Step 4: TRAIN
- Splits data: 67% training (596 samples), 33% testing (295 samples)
- Encodes categorical variables
- Trains Decision Tree Classifier
- Displays model coefficients and feature importance

##### Step 5: TEST
- Calculates training metrics (Accuracy, Precision, Recall, F1)
- Calculates testing metrics
- Shows confusion matrix
- Displays actual vs predicted plots
- Analyzes model performance

##### Step 6: DEPLOY
- Creates prediction examples
- Makes predictions on sample passenger data
- Displays prediction results with interpretation

#### 3. Modify and Experiment
- Edit cell code and re-run
- Add new cells for custom analysis
- Create visualizations
- Save modified notebook

### Using Python Script

#### 1. Run the Script
```bash
python BITS_AIML_Titanic_Jan3rd2026.py
```

#### 2. View Console Output
- Dataset information displays in terminal
- Statistical summaries print to console
- Model metrics display as text
- Analysis results show in terminal

#### 3. View Visualizations
- Matplotlib windows open automatically
- Multiple plots display sequentially
- Close each window to continue execution
- All visualizations appear during script execution

#### 4. Batch Processing
- Script runs completely without interaction
- All results saved to console output
- Suitable for automated analysis
- Can redirect output to file: `python script.py > output.txt`

---

## 📊 Dataset Information

### Source
- **URL:** https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv
- **Format:** CSV
- **Target Variable:** Survived (0 = Did not survive, 1 = Survived)

### Features
- **PassengerId:** Unique passenger identifier
- **Pclass:** Passenger class (1, 2, or 3)
- **Name:** Passenger name
- **Sex:** Gender (male/female)
- **Age:** Age in years (177 missing values)
- **SibSp:** Number of siblings/spouses aboard
- **Parch:** Number of parents/children aboard
- **Ticket:** Ticket number
- **Fare:** Ticket fare
- **Cabin:** Cabin number (687 missing values)
- **Embarked:** Port of embarkation (C, Q, S) (2 missing values)

### Data Characteristics
- 891 observations
- 12 features (mix of numeric and categorical)
- 38.38% survival rate
- Class-based survival disparity
- Gender-based survival disparity

---

## 🔧 Decision Tree Classifier Architecture

### Model Configuration

```python
DecisionTreeClassifier(
    random_state=3,           # Reproducibility
    max_depth=5,              # Prevent overfitting
    min_samples_split=10,     # Minimum samples to split
    min_samples_leaf=5        # Minimum samples in leaf
)
```

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| random_state | 3 | Reproducibility across runs |
| max_depth | 5 | Prevent overfitting |
| min_samples_split | 10 | Reduce noise |
| min_samples_leaf | 5 | Ensure leaf stability |

### Helper Functions

#### `load_and_analyze_data()`
- Loads dataset from GitHub
- Performs initial analysis
- Returns DataFrame and analysis dictionary

#### `clean_data(df)`
- Handles missing values strategically
- Removes duplicates
- Returns cleaned DataFrame and statistics

#### `create_visualizations(df)`
- Creates multiple visualization types
- Analyzes survival patterns
- Returns visualization objects

#### `train_model(df, target_col)`
- Prepares features and target
- Splits data (67/33 with random_state=3)
- Trains Decision Tree Classifier
- Returns model and training info

#### `evaluate_model(model, X_train, X_test, y_train, y_test)`
- Calculates classification metrics
- Creates visualization images
- Returns comprehensive metrics dictionary

#### `deploy_predictions(model, scaler, feature_cols)`
- Makes predictions on sample data
- Displays prediction results
- Shows prediction interpretation

---

## 📈 Expected Performance

### Typical Metrics (on Titanic dataset)
- **Accuracy:** 78-82% (78-82% of predictions correct)
- **Precision:** 70-75% (of predicted survivors, 70-75% actually survived)
- **Recall:** 70-75% (of actual survivors, 70-75% were correctly predicted)
- **F1-Score:** 72-76% (harmonic mean of precision and recall)

### Confusion Matrix Interpretation
```
                Predicted Did Not Survive    Predicted Survived
Actually Did Not Survive:  TN (True Negatives)    FP (False Positives)
Actually Survived:         FN (False Negatives)   TP (True Positives)
```

### Key Insights
- "Women and children first" policy clearly evident in predictions
- Class-based survival patterns captured by model
- Age affects survival chances
- Model achieves good balance between precision and recall

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Install missing packages
```bash
pip install -r requirements.txt
```

### Issue: "Connection Error" when loading dataset
**Solution:** Check internet connection and GitHub availability
```bash
# Test connection
curl https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv
```

### Issue: Jupyter not found
**Solution:** Install Jupyter
```bash
pip install jupyter notebook
```

### Issue: Matplotlib display error
**Solution:** For Jupyter, use magic command
```python
%matplotlib inline
```

### Issue: "Kernel dead" or "Kernel restarting" (Jupyter)
**Solution:** Restart kernel
- Kernel menu → Restart
- Or stop Jupyter and start again

### Issue: Slow performance
**Solution:**
- Check internet connection
- Reduce visualization resolution
- Close other applications

### Issue: Data loading fails
**Solution:**
- Verify internet connection
- Check GitHub URL accessibility
- Ensure pandas is properly installed

---

## 🚀 Deployment Options

### Option 1: Jupyter Hub (Multi-user)

1. **Install JupyterHub:**
```bash
pip install jupyterhub
```

2. **Configure and run:**
```bash
jupyterhub
```

3. **Access at:** `http://localhost:8000`

### Option 2: Voila (Interactive Dashboard)

1. **Install Voila:**
```bash
pip install voila
```

2. **Run notebook as app:**
```bash
voila BITS_AIML_Titanic_Jan3rd2026.ipynb
```

3. **Access at:** `http://localhost:8866`

### Option 3: Docker Deployment

1. **Create Dockerfile:**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
```

2. **Build and run:**
```bash
docker build -t titanic-notebook .
docker run -p 8888:8888 titanic-notebook
```

### Option 4: Binder (Cloud-based)

1. **Create environment.yml:**
```yaml
name: titanic
channels:
  - conda-forge
dependencies:
  - python=3.9
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter
```

2. **Push to GitHub**

3. **Create Binder link:**
```
https://mybinder.org/v2/gh/USERNAME/REPO/main?filepath=titanic/BITS_AIML_Titanic_Jan3rd2026.ipynb
```

### Option 5: Google Colab (No Setup Required)

1. **Upload notebook to Google Drive**
2. **Open with Google Colab**
3. **Run cells directly in cloud**
4. **No local installation needed**

---

## 📚 Learning Resources

### Jupyter Documentation
- https://jupyter.org/documentation

### Scikit-learn Documentation
- https://scikit-learn.org/stable/

### Decision Trees
- https://scikit-learn.org/stable/modules/tree.html

### Pandas Documentation
- https://pandas.pydata.org/docs/

### Matplotlib Documentation
- https://matplotlib.org/

---

## 🎓 Educational Value

This project demonstrates:
- ✓ Complete ML classification workflow
- ✓ Data preprocessing techniques
- ✓ Exploratory data analysis
- ✓ Decision tree classifier implementation
- ✓ Model evaluation and metrics
- ✓ Feature importance analysis
- ✓ Real-world prediction deployment
- ✓ Jupyter notebook best practices

---

## 📄 File Structure

```
BITS_Hackathon/titanic/
├── BITS_AIML_Titanic_Jan3rd2026.ipynb      # Jupyter notebook
├── BITS_AIML_Titanic_Jan3rd2026.py         # Python script
├── requirements.txt                         # Python dependencies
├── TITANIC_JUPYTER_DEPLOYMENT.md            # This deployment guide
├── README.md                                # Project README
└── SETUP.md                                 # Setup instructions
```

---

## 🔐 Security Considerations

- **Input Validation:** All inputs validated before processing
- **Error Handling:** Comprehensive error handling
- **Data Privacy:** No data stored permanently
- **Code Safety:** No arbitrary code execution
- **Dependency Security:** All packages from official sources

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review Jupyter documentation
3. Check dataset availability on GitHub
4. Verify Python and package versions
5. Review scikit-learn documentation

---

## 📜 License

This project uses publicly available datasets and open-source libraries.

---

## ✅ Quick Start Checklist

Before running the application:
- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Internet connection available (for GitHub dataset)
- [ ] Jupyter installed (`pip install jupyter`)
- [ ] All files in correct directory

---

**Last Updated:** Jan 4, 2026
**Version:** 1.0.0
**Status:** ✅ Production Ready

---

## 🎯 Next Steps

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Start Jupyter:** `jupyter notebook`
3. **Open notebook:** Click on `BITS_AIML_Titanic_Jan3rd2026.ipynb`
4. **Execute cells:** Run cells sequentially using Shift+Enter
5. **Experiment:** Modify code and observe results
6. **Deploy:** Use Voila, JupyterHub, or Docker for production

Enjoy analyzing the Titanic dataset! 🎉
