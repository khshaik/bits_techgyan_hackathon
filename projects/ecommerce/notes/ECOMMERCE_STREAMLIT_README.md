# E-Commerce Customer Analysis - Streamlit Application

## 📊 Project Overview

This Streamlit application performs comprehensive analysis on E-commerce customer data using Linear Regression to predict customer spending patterns. The application implements a complete machine learning workflow with interactive visualizations and real-time predictions.

**Live Dataset Source:** https://github.com/araj2/customer-database/blob/master/Ecommerce%20Customers.csv

---

## 🎯 Workflow Steps

### 1. **ANALYZE** - Dataset Exploration
- Load E-commerce customer dataset from GitHub
- Display dataset dimensions and structure
- Show column information and data types
- Generate statistical summaries
- Identify missing values and data quality issues

### 2. **CLEAN** - Data Preprocessing
- Handle missing values (removal or imputation)
- Remove duplicate records
- Validate data integrity
- Prepare data for modeling
- Display cleaning statistics

### 3. **VISUALIZE** - Data Exploration
- Distribution of target variable (Yearly Amount Spent)
- Correlation heatmap of all features
- Feature distributions
- Scatter plots showing relationships with target variable
- Interactive visualizations for insights

### 4. **TRAIN** - Model Development
- Prepare features and target variable
- Split data: 67% training, 33% testing (random_state=3)
- Standardize features using StandardScaler
- Train Linear Regression model
- Display model coefficients

### 5. **TEST** - Model Evaluation
- Calculate comprehensive performance metrics:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**
  - **R² Score**
- Generate confusion matrix and classification metrics
- Visualize actual vs predicted values
- Analyze residuals
- Compare training vs testing performance

### 6. **DEPLOY** - Make Predictions
- Interactive prediction interface
- Input customer characteristics
- Real-time spending prediction
- Display prediction results with confidence

---

## 📋 Performance Metrics

The application calculates and displays:

### Regression Metrics
- **MSE (Mean Squared Error):** Average squared difference between predicted and actual values
- **RMSE (Root Mean Squared Error):** Square root of MSE, in same units as target
- **MAE (Mean Absolute Error):** Average absolute difference between predictions and actuals
- **R² Score:** Proportion of variance explained by the model (0-1 scale)

### Visualizations
- Actual vs Predicted scatter plots
- Residual distributions
- Feature importance based on coefficients
- Correlation heatmaps

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

Or install individually:

```bash
pip install streamlit==1.28.1
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
```

### Step 2: Run the Streamlit Application

```bash
streamlit run ecommerce_streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📊 Dataset Information

### Source
- **URL:** https://github.com/araj2/customer-database/blob/master/Ecommerce%20Customers.csv
- **Format:** CSV
- **Target Variable:** Yearly Amount Spent

### Features
The dataset typically includes:
- **Avg. Session Length:** Average session duration in minutes
- **Time on App:** Time spent on mobile app in minutes
- **Time on Website:** Time spent on website in minutes
- **Length of Membership:** Customer membership duration in years
- **Yearly Amount Spent:** Target variable (customer spending)

### Data Characteristics
- Numeric features for customer behavior metrics
- Continuous target variable for spending prediction
- No categorical variables (all numeric)

---

## 🚀 How to Use the Application

### 1. View Dataset Analysis
- The application automatically loads and displays the dataset
- Review dataset dimensions, columns, and statistics
- Check for missing values and data quality

### 2. Explore Visualizations
- Scroll through interactive visualizations
- Analyze feature distributions
- Review correlation matrix
- Examine relationships with target variable

### 3. Review Model Performance
- Check training and testing metrics
- Compare MSE, RMSE, MAE, and R² scores
- Analyze actual vs predicted plots
- Review residual distributions

### 4. Make Predictions
- Enter customer characteristics in the input fields
- Click "Predict Spending" button
- View predicted yearly spending amount
- See input summary

---

## 📈 Model Details

### Algorithm
**Linear Regression**
- Simple, interpretable model
- Suitable for continuous target variable prediction
- Fast training and inference
- Provides feature coefficients for interpretability

### Data Split
- **Training Data:** 67% (random_state=3)
- **Testing Data:** 33% (random_state=3)
- Fixed random state ensures reproducibility

### Feature Scaling
- **StandardScaler** applied to normalize features
- Improves model convergence
- Ensures fair feature comparison

### Model Parameters
```python
LinearRegression(
    fit_intercept=True,
    normalize=False,
    copy_X=True,
    n_jobs=None
)
```

---

## 📊 Expected Performance

### Typical Metrics (on E-commerce dataset)
- **R² Score:** 0.98-0.99 (98-99% variance explained)
- **RMSE:** $100-500 (depending on spending scale)
- **MAE:** $50-300 (average prediction error)

### Interpretation
- High R² indicates strong predictive power
- Low RMSE/MAE indicates accurate predictions
- Model suitable for customer spending forecasting

---

## 🔍 Key Features

### Interactive Dashboard
- Real-time data loading from GitHub
- Dynamic visualizations
- Responsive layout
- Professional styling

### Comprehensive Analysis
- Complete data exploration
- Detailed preprocessing steps
- Multiple visualization types
- Thorough model evaluation

### User-Friendly Interface
- Clear section headers
- Metric displays
- Data tables
- Input forms for predictions

### Production-Ready Code
- Comprehensive documentation
- Error handling
- Modular functions
- Clean code structure

---

## 📝 Code Structure

```
ecommerce_streamlit_app.py
├── Imports and Configuration
├── Step 1: load_and_analyze_data()
├── Step 2: clean_data()
├── Step 3: visualize_data()
├── Step 4: train_model()
├── Step 5: evaluate_model()
├── Step 6: deploy_predictions()
└── main()
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Install missing packages using pip
```bash
pip install -r requirements_streamlit.txt
```

### Issue: "Connection Error" when loading dataset
**Solution:** Check internet connection and GitHub availability
```bash
# Test connection
curl https://raw.githubusercontent.com/araj2/customer-database/master/Ecommerce%20Customers.csv
```

### Issue: Streamlit not found
**Solution:** Install Streamlit globally
```bash
pip install streamlit --upgrade
```

### Issue: Slow performance
**Solution:** Clear Streamlit cache
```bash
streamlit cache clear
```

---

## 📚 Learning Resources

### Streamlit Documentation
- https://docs.streamlit.io/

### Scikit-learn Documentation
- https://scikit-learn.org/stable/

### Linear Regression
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

### Pandas Documentation
- https://pandas.pydata.org/docs/

---

## 🎓 Educational Value

This application demonstrates:
- ✓ Complete ML workflow implementation
- ✓ Data preprocessing techniques
- ✓ Exploratory data analysis
- ✓ Model training and evaluation
- ✓ Interactive web application development
- ✓ Performance metrics calculation
- ✓ Real-world prediction deployment

---

## 📄 File Structure

```
FUNNYAPP/
├── ecommerce_streamlit_app.py      # Main Streamlit application
├── requirements_streamlit.txt       # Python dependencies
├── ECOMMERCE_STREAMLIT_README.md   # This file
└── price_prediction.py              # Linear Regression example
```

---

## 🔐 Data Privacy

- Dataset loaded from public GitHub repository
- No data stored locally
- No personal information collected
- Predictions made locally on user's machine

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review Streamlit documentation
3. Check dataset availability on GitHub
4. Verify Python and package versions

---

## 📜 License

This project uses publicly available datasets and open-source libraries.

---

## ✅ Checklist

Before running the application:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements_streamlit.txt`)
- [ ] Internet connection available (for GitHub dataset)
- [ ] Streamlit installed (`pip install streamlit`)

---

**Last Updated:** Jan 3, 2026
**Version:** 1.0.0
**Status:** ✅ Production Ready
