"""
ADVERTISING DATASET ANALYSIS - FLASK APPLICATION

Project Overview:
This Flask application performs comprehensive analysis on Advertising dataset
using Linear Regression to predict sales based on advertising spend across channels.

Workflow Steps:
1. ANALYZE - Explore dataset structure and characteristics
2. CLEAN - Handle missing values and prepare data
3. VISUALIZE - Create interactive visualizations
4. TRAIN - Build and train Linear Regression model (67% training data)
5. TEST - Evaluate model performance with comprehensive metrics
6. DEPLOY - Make predictions on new advertising spend data

Dataset Source:
https://github.com/erkansirin78/datasets/blob/master/Advertising.csv

Target Variable: Sales (revenue prediction based on advertising spend)
Features: TV, Radio, Newspaper advertising spend
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             r2_score, accuracy_score)
import warnings
import io
import base64
from urllib.request import urlopen

warnings.filterwarnings('ignore')

# Initialize Flask application
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global variables to store model and data
model = None
scaler = None
feature_cols = None
df_clean = None
metrics = None

# ============================================================================
# STEP 1: ANALYZE - Load and Explore Dataset
# ============================================================================

def load_and_analyze_data():
    """
    Load Advertising dataset from GitHub and perform initial analysis.
    
    Returns:
    - DataFrame: Loaded advertising data
    """
    # Load dataset from GitHub
    url = "https://raw.githubusercontent.com/erkansirin78/datasets/master/Advertising.csv"
    
    try:
        df = pd.read_csv(url)
        
        # Remove unnamed index column if present
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        analysis = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'column_info': [
                {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'non_null': int(df[col].count()),
                    'null_count': int(df[col].isnull().sum())
                }
                for col in df.columns
            ],
            'statistics': df.describe().to_dict(),
            'first_rows': df.head(10).to_dict('records')
        }
        
        return df, analysis
        
    except Exception as e:
        return None, {'error': str(e)}

# ============================================================================
# STEP 2: CLEAN - Data Preprocessing
# ============================================================================

def clean_data(df):
    """
    Clean and preprocess the dataset.
    
    Steps:
    - Handle missing values
    - Remove duplicates
    - Prepare data for modeling
    
    Parameters:
    - df: Raw DataFrame
    
    Returns:
    - DataFrame: Cleaned data
    - dict: Cleaning statistics
    """
    original_rows = len(df)
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    df_clean = df.dropna()
    rows_removed_missing = original_rows - len(df_clean)
    
    # Remove duplicates
    duplicates = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    
    cleaning_stats = {
        'original_rows': original_rows,
        'rows_after_cleaning': len(df_clean),
        'rows_removed': original_rows - len(df_clean),
        'missing_values_found': int(missing_before),
        'rows_removed_missing': int(rows_removed_missing),
        'duplicates_removed': int(duplicates),
        'columns': len(df_clean.columns),
        'column_names': list(df_clean.columns)
    }
    
    return df_clean, cleaning_stats

# ============================================================================
# STEP 3: VISUALIZE - Data Visualization
# ============================================================================

def create_visualizations(df):
    """
    Create visualizations for data exploration.
    
    Parameters:
    - df: Cleaned DataFrame
    
    Returns:
    - dict: Base64 encoded images
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'Sales' if 'Sales' in numeric_cols else numeric_cols[-1]
    
    visualizations = {}
    
    # Visualization 1: Target Variable Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df[target_col], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title(f'{target_col} Distribution', fontweight='bold')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Frequency')
    
    axes[1].boxplot(df[target_col], vert=True)
    axes[1].set_title(f'{target_col} Box Plot', fontweight='bold')
    axes[1].set_ylabel(target_col)
    
    plt.tight_layout()
    visualizations['distribution'] = fig_to_base64(fig)
    plt.close(fig)
    
    # Visualization 2: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    visualizations['correlation'] = fig_to_base64(fig)
    plt.close(fig)
    
    # Visualization 3: Feature Distributions
    feature_cols = [col for col in numeric_cols if col != target_col][:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col], bins=20, color='lightgreen', edgecolor='black')
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    visualizations['features'] = fig_to_base64(fig)
    plt.close(fig)
    
    # Visualization 4: Scatter plots with target
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, col in enumerate(feature_cols):
        axes[idx].scatter(df[col], df[target_col], alpha=0.5, color='purple')
        axes[idx].set_title(f'{col} vs {target_col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel(target_col)
    
    plt.tight_layout()
    visualizations['scatter'] = fig_to_base64(fig)
    plt.close(fig)
    
    return visualizations, target_col, feature_cols

# ============================================================================
# STEP 4: TRAIN - Model Training
# ============================================================================

def train_model(df, target_col):
    """
    Prepare data and train Linear Regression model.
    
    Parameters:
    - df: Cleaned DataFrame
    - target_col: Target variable column name
    
    Returns:
    - Tuple: (model, X_train, X_test, y_train, y_test, scaler, feature_cols)
    """
    # Identify numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    # Prepare features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data: 67% training, 33% testing with random_state=3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=3
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    training_info = {
        'total_samples': len(X),
        'num_features': len(feature_cols),
        'feature_names': feature_cols,
        'target_variable': target_col,
        'training_samples': len(X_train),
        'training_percentage': round(len(X_train)/len(X)*100, 1),
        'testing_samples': len(X_test),
        'testing_percentage': round(len(X_test)/len(X)*100, 1),
        'model_intercept': float(model.intercept_),
        'model_coefficients': {feature_cols[i]: float(model.coef_[i]) 
                               for i in range(len(feature_cols))}
    }
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, training_info

# ============================================================================
# STEP 5: TEST - Model Evaluation
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance with comprehensive metrics.
    
    Parameters:
    - model: Trained Linear Regression model
    - X_train, X_test: Training and testing features
    - y_train, y_test: Training and testing targets
    
    Returns:
    - dict: Performance metrics and visualizations
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Create visualizations
    visualizations = {}
    
    # Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_train, y_train_pred, alpha=0.5, color='blue')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                 'r--', lw=2)
    axes[0].set_title('Training: Actual vs Predicted', fontweight='bold')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test, y_test_pred, alpha=0.5, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2)
    axes[1].set_title('Testing: Actual vs Predicted', fontweight='bold')
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    visualizations['actual_vs_predicted'] = fig_to_base64(fig)
    plt.close(fig)
    
    # Residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    axes[0].hist(train_residuals, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Training Residuals', fontweight='bold')
    axes[0].set_xlabel('Residuals')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    axes[1].hist(test_residuals, bins=30, color='lightgreen', edgecolor='black')
    axes[1].set_title('Testing Residuals', fontweight='bold')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    visualizations['residuals'] = fig_to_base64(fig)
    plt.close(fig)
    
    metrics_data = {
        'training': {
            'mse': round(train_mse, 6),
            'rmse': round(train_rmse, 6),
            'mae': round(train_mae, 6),
            'r2_score': round(train_r2, 6),
            'r2_percentage': round(train_r2 * 100, 2)
        },
        'testing': {
            'mse': round(test_mse, 6),
            'rmse': round(test_rmse, 6),
            'mae': round(test_mae, 6),
            'r2_score': round(test_r2, 6),
            'r2_percentage': round(test_r2 * 100, 2)
        },
        'visualizations': visualizations
    }
    
    return metrics_data

# ============================================================================
# STEP 6: DEPLOY - Make Predictions
# ============================================================================

def make_prediction(input_values, model, scaler, feature_cols):
    """
    Make prediction on new data.
    
    Parameters:
    - input_values: Dictionary of feature values
    - model: Trained model
    - scaler: Fitted scaler
    - feature_cols: Feature column names
    
    Returns:
    - float: Predicted value
    """
    input_array = np.array([[input_values[col] for col in feature_cols]])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    return round(prediction, 2)

# ============================================================================
# Helper Functions
# ============================================================================

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string."""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['GET'])
def api_analyze():
    """Analyze dataset."""
    df, analysis = load_and_analyze_data()
    if df is None:
        return jsonify({'error': analysis['error']}), 400
    
    return jsonify(analysis)

@app.route('/api/clean', methods=['GET'])
def api_clean():
    """Clean dataset."""
    global df_clean
    df, _ = load_and_analyze_data()
    df_clean, cleaning_stats = clean_data(df)
    
    return jsonify(cleaning_stats)

@app.route('/api/visualize', methods=['GET'])
def api_visualize():
    """Create visualizations."""
    if df_clean is None:
        return jsonify({'error': 'Data not cleaned yet'}), 400
    
    visualizations, target_col, feature_cols = create_visualizations(df_clean)
    
    return jsonify({
        'visualizations': visualizations,
        'target_column': target_col,
        'feature_columns': feature_cols
    })

@app.route('/api/train', methods=['GET'])
def api_train():
    """Train model."""
    global model, scaler, feature_cols
    
    if df_clean is None:
        return jsonify({'error': 'Data not cleaned yet'}), 400
    
    # Identify target column
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'Sales' if 'Sales' in numeric_cols else numeric_cols[-1]
    
    model, X_train, X_test, y_train, y_test, scaler, feature_cols, training_info = train_model(df_clean, target_col)
    
    return jsonify(training_info)

@app.route('/api/test', methods=['GET'])
def api_test():
    """Test model."""
    global model, scaler, feature_cols, metrics
    
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    # Prepare data for testing
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'Sales' if 'Sales' in numeric_cols else numeric_cols[-1]
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=3
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    
    return jsonify(metrics)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Make prediction."""
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    data = request.get_json()
    
    try:
        input_values = {col: float(data.get(col, 0)) for col in feature_cols}
        prediction = make_prediction(input_values, model, scaler, feature_cols)
        
        return jsonify({
            'prediction': prediction,
            'input_values': input_values,
            'features': feature_cols
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    """Dashboard page."""
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
