# Advertising Dataset Analysis - Flask Application Deployment Guide

## 📊 Project Overview

This Flask application performs comprehensive analysis on Advertising dataset using Linear Regression to predict sales based on advertising spend across different channels (TV, Radio, Newspaper).

**Live Dataset Source:** https://github.com/erkansirin78/datasets/blob/master/Advertising.csv

---

## 🎯 Workflow Steps

### 1. **ANALYZE** - Dataset Exploration
- Load Advertising dataset from GitHub
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
- Distribution of target variable (Sales)
- Correlation heatmap of all features
- Feature distributions (TV, Radio, Newspaper)
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
- Input advertising spend values
- Real-time sales prediction
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
- Git (optional, for cloning repository)

### Step 1: Install Dependencies

```bash
pip install -r requirements_flask.txt
```

Or install individually:

```bash
pip install Flask==2.3.2
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install Werkzeug==2.3.6
```

### Step 2: Project Structure

Ensure your project structure looks like this:

```
FUNNYAPP/
├── advertising_flask_app.py          # Main Flask application
├── requirements_flask.txt             # Python dependencies
├── templates/
│   └── index.html                     # HTML template
├── advertising_analysis.ipynb         # Jupyter notebook
└── ADVERTISING_FLASK_DEPLOYMENT.md   # This file
```

### Step 3: Run the Flask Application

```bash
python advertising_flask_app.py
```

The application will start at `http://localhost:5000`

---

## 🚀 How to Use the Application

### 1. Access the Dashboard
- Open your browser and navigate to `http://localhost:5000`
- You'll see the Advertising Analysis Dashboard with 6 workflow steps

### 2. Run Analysis Steps

Click on each step button to execute:

#### Step 1: ANALYZE
- Loads dataset from GitHub
- Displays dataset dimensions (200 rows, 4 columns)
- Shows column information and statistics
- Identifies missing values (typically 0)

#### Step 2: CLEAN
- Removes rows with missing values
- Removes duplicate records
- Displays cleaning statistics
- Shows final cleaned dataset dimensions

#### Step 3: VISUALIZE
- Distribution plot of Sales (target variable)
- Correlation heatmap showing relationships
- Feature distributions (TV, Radio, Newspaper)
- Scatter plots showing feature vs target relationships

#### Step 4: TRAIN
- Splits data: 67% training (134 samples), 33% testing (66 samples)
- Standardizes features using StandardScaler
- Trains Linear Regression model
- Displays model coefficients for each feature

#### Step 5: TEST
- Calculates training metrics (MSE, RMSE, MAE, R²)
- Calculates testing metrics (MSE, RMSE, MAE, R²)
- Shows actual vs predicted scatter plots
- Displays residual distributions

#### Step 6: PREDICT
- Enter advertising spend values (TV, Radio, Newspaper)
- Click "Predict Sales" button
- View predicted sales amount
- See input summary

---

## 📊 Dataset Information

### Source
- **URL:** https://github.com/erkansirin78/datasets/blob/master/Advertising.csv
- **Format:** CSV
- **Target Variable:** Sales (revenue in thousands)

### Features
- **TV:** TV advertising spend (in thousands of dollars)
- **Radio:** Radio advertising spend (in thousands of dollars)
- **Newspaper:** Newspaper advertising spend (in thousands of dollars)
- **Sales:** Sales revenue (in thousands of dollars)

### Data Characteristics
- 200 observations
- 4 numeric features (all continuous)
- No categorical variables
- Typically no missing values

---

## 🔧 Flask Application Architecture

### Routes

#### GET `/`
- Home page with workflow interface
- Displays 6 step buttons
- Main dashboard

#### GET `/api/analyze`
- Returns dataset analysis
- Includes: rows, columns, missing values, column info, statistics

#### GET `/api/clean`
- Returns cleaning statistics
- Includes: rows removed, duplicates removed, final dimensions

#### GET `/api/visualize`
- Returns base64 encoded visualization images
- Includes: distribution, correlation, features, scatter plots

#### GET `/api/train`
- Returns training information
- Includes: feature names, coefficients, intercept, data split info

#### GET `/api/test`
- Returns model evaluation metrics
- Includes: MSE, RMSE, MAE, R² for training and testing
- Includes: visualization images

#### POST `/api/predict`
- Accepts JSON with feature values
- Returns predicted sales value
- Includes: input values, features used

### Helper Functions

#### `load_and_analyze_data()`
- Loads dataset from GitHub
- Performs initial analysis
- Returns DataFrame and analysis dictionary

#### `clean_data(df)`
- Handles missing values
- Removes duplicates
- Returns cleaned DataFrame and statistics

#### `create_visualizations(df)`
- Creates 4 types of visualizations
- Returns base64 encoded images
- Identifies target and feature columns

#### `train_model(df, target_col)`
- Prepares features and target
- Splits data (67/33 with random_state=3)
- Trains Linear Regression model
- Returns model, scalers, and training info

#### `evaluate_model(model, X_train, X_test, y_train, y_test)`
- Calculates performance metrics
- Creates visualization images
- Returns comprehensive metrics dictionary

#### `make_prediction(input_values, model, scaler, feature_cols)`
- Makes prediction on new data
- Scales input using fitted scaler
- Returns predicted value

#### `fig_to_base64(fig)`
- Converts matplotlib figure to base64
- Used for embedding images in HTML

---

## 📈 Expected Performance

### Typical Metrics (on Advertising dataset)
- **R² Score:** 0.89-0.91 (89-91% variance explained)
- **RMSE:** 1.5-2.0 (in thousands of dollars)
- **MAE:** 1.2-1.5 (in thousands of dollars)

### Interpretation
- High R² indicates strong predictive power
- Low RMSE/MAE indicates accurate predictions
- Model suitable for sales forecasting

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Install missing packages
```bash
pip install -r requirements_flask.txt
```

### Issue: "Connection Error" when loading dataset
**Solution:** Check internet connection and GitHub availability
```bash
# Test connection
curl https://raw.githubusercontent.com/erkansirin78/datasets/master/Advertising.csv
```

### Issue: Flask not found
**Solution:** Install Flask globally
```bash
pip install Flask --upgrade
```

### Issue: Port 5000 already in use
**Solution:** Use different port
```bash
# Edit advertising_flask_app.py
# Change: app.run(debug=True, port=5000)
# To: app.run(debug=True, port=5001)
```

### Issue: Slow performance
**Solution:** 
- Check internet connection
- Reduce visualization resolution
- Use smaller dataset sample

---

## 🚀 Deployment to Production

### Option 1: Heroku Deployment

1. **Create Procfile:**
```
web: gunicorn advertising_flask_app:app
```

2. **Create runtime.txt:**
```
python-3.9.16
```

3. **Install gunicorn:**
```bash
pip install gunicorn
```

4. **Deploy to Heroku:**
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Option 2: AWS Deployment

1. **Create EC2 instance**
2. **Install dependencies**
3. **Use Nginx as reverse proxy**
4. **Use Gunicorn as application server**
5. **Configure SSL/TLS**

### Option 3: Docker Deployment

1. **Create Dockerfile:**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements_flask.txt .
RUN pip install -r requirements_flask.txt
COPY . .
CMD ["python", "advertising_flask_app.py"]
```

2. **Build and run:**
```bash
docker build -t advertising-app .
docker run -p 5000:5000 advertising-app
```

---

## 📚 Learning Resources

### Flask Documentation
- https://flask.palletsprojects.com/

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
- ✓ Web application development with Flask
- ✓ Performance metrics calculation
- ✓ Real-world prediction deployment
- ✓ RESTful API design

---

## 📄 File Structure

```
FUNNYAPP/
├── advertising_flask_app.py              # Main Flask application
├── requirements_flask.txt                # Python dependencies
├── templates/
│   └── index.html                        # HTML template with CSS & JS
├── advertising_analysis.ipynb            # Jupyter notebook version
├── ADVERTISING_FLASK_DEPLOYMENT.md      # This deployment guide
└── README.md                             # Project README
```

---

## 🔐 Security Considerations

- **Input Validation:** All inputs validated before processing
- **Error Handling:** Comprehensive error handling with safe messages
- **CORS:** Configured for local development
- **Data Privacy:** No data stored permanently
- **API Security:** No sensitive information exposed

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review Flask documentation
3. Check dataset availability on GitHub
4. Verify Python and package versions

---

## 📜 License

This project uses publicly available datasets and open-source libraries.

---

## ✅ Quick Start Checklist

Before running the application:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements_flask.txt`)
- [ ] Internet connection available (for GitHub dataset)
- [ ] Port 5000 available
- [ ] Flask installed (`pip install flask`)

---

**Last Updated:** Jan 3, 2026
**Version:** 1.0.0
**Status:** ✅ Production Ready

---

## 🎯 Next Steps

1. **Run the Flask app:** `python advertising_flask_app.py`
2. **Open browser:** Navigate to `http://localhost:5000`
3. **Click workflow steps:** Execute analysis steps sequentially
4. **Make predictions:** Enter advertising spend values and predict sales
5. **Deploy:** Follow production deployment options above

Enjoy analyzing the Advertising dataset! 🎉
