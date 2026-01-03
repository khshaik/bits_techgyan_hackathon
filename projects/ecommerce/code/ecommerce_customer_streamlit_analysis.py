"""
E-COMMERCE CUSTOMER ANALYSIS - STREAMLIT APPLICATION

Project Overview:
This Streamlit application performs comprehensive analysis on E-commerce customer data
using Linear Regression to predict customer spending based on their characteristics.

Workflow Steps:
1. ANALYZE - Explore dataset structure and characteristics
2. CLEAN - Handle missing values and prepare data
3. VISUALIZE - Create interactive visualizations
4. TRAIN - Build and train Linear Regression model (67% training data)
5. TEST - Evaluate model performance with comprehensive metrics
6. DEPLOY - Make predictions on new customer data

Dataset Source:
https://github.com/araj2/customer-database/blob/master/Ecommerce%20Customers.csv

Target Variable: Yearly Amount Spent (customer spending prediction)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             r2_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="E-commerce Customer Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# STEP 1: ANALYZE - Load and Explore Dataset
# ============================================================================

def load_and_analyze_data():
    """
    Load E-commerce customer dataset from GitHub and perform initial analysis.
    
    Returns:
    - DataFrame: Loaded customer data
    """
    st.header("📊 Step 1: ANALYZE - Dataset Exploration")
    
    # Load dataset from GitHub
    url = "https://raw.githubusercontent.com/araj2/customer-database/master/Ecommerce%20Customers.csv"
    
    try:
        df = pd.read_csv(url)
        st.success("✓ Dataset loaded successfully!")
        
        # Display dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Types", len(df.dtypes.unique()))
        
        # Display first few rows
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Display column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Display statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# ============================================================================
# STEP 2: CLEAN - Data Preprocessing
# ============================================================================

def clean_data(df):
    """
    Clean and preprocess the dataset.
    
    Steps:
    - Handle missing values
    - Remove duplicates
    - Select relevant features
    - Prepare data for modeling
    
    Parameters:
    - df: Raw DataFrame
    
    Returns:
    - DataFrame: Cleaned data ready for analysis
    """
    st.header("🧹 Step 2: CLEAN - Data Preprocessing")
    
    # Check missing values
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum()
    
    if missing_values.sum() > 0:
        st.warning(f"Found {missing_values.sum()} missing values")
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing Percentage': (missing_values.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        st.dataframe(missing_df, use_container_width=True)
        
        # Handle missing values
        df_clean = df.dropna()
        st.success(f"✓ Removed {len(df) - len(df_clean)} rows with missing values")
    else:
        df_clean = df.copy()
        st.success("✓ No missing values found")
    
    # Check duplicates
    st.subheader("Duplicate Records Analysis")
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        st.warning(f"Found {duplicates} duplicate records")
        df_clean = df_clean.drop_duplicates()
        st.success(f"✓ Removed {duplicates} duplicate records")
    else:
        st.success("✓ No duplicate records found")
    
    # Display cleaned dataset info
    st.subheader("Cleaned Dataset Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows After Cleaning", len(df_clean))
    with col2:
        st.metric("Columns", len(df_clean.columns))
    with col3:
        st.metric("Rows Removed", len(df) - len(df_clean))
    
    return df_clean

# ============================================================================
# STEP 3: VISUALIZE - Data Visualization
# ============================================================================

def visualize_data(df):
    """
    Create interactive visualizations for data exploration.
    
    Visualizations:
    - Distribution of target variable
    - Correlation heatmap
    - Feature distributions
    - Relationship with target variable
    
    Parameters:
    - df: Cleaned DataFrame
    """
    st.header("📈 Step 3: VISUALIZE - Data Exploration")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for visualization")
        return
    
    # Target variable (usually the last numeric column or 'Yearly Amount Spent')
    target_col = 'Yearly Amount Spent' if 'Yearly Amount Spent' in numeric_cols else numeric_cols[-1]
    
    # Visualization 1: Target Variable Distribution
    st.subheader(f"1. Distribution of {target_col}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df[target_col], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title(f'{target_col} Distribution', fontweight='bold')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Frequency')
    
    axes[1].boxplot(df[target_col], vert=True)
    axes[1].set_title(f'{target_col} Box Plot', fontweight='bold')
    axes[1].set_ylabel(target_col)
    
    st.pyplot(fig)
    
    # Visualization 2: Correlation Heatmap
    st.subheader("2. Correlation Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    st.pyplot(fig)
    
    # Visualization 3: Feature Distributions
    st.subheader("3. Feature Distributions")
    feature_cols = [col for col in numeric_cols if col != target_col][:4]  # First 4 features
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col], bins=20, color='lightgreen', edgecolor='black')
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Visualization 4: Scatter plots with target variable
    st.subheader("4. Feature vs Target Variable")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_cols):
        axes[idx].scatter(df[col], df[target_col], alpha=0.5, color='purple')
        axes[idx].set_title(f'{col} vs {target_col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel(target_col)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return target_col, feature_cols

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
    - Tuple: (model, X_train, X_test, y_train, y_test, scaler)
    """
    st.header("🤖 Step 4: TRAIN - Model Development")
    
    # Identify numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    # Prepare features and target
    X = df[feature_cols]
    y = df[target_col]
    
    st.subheader("Data Preparation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(X))
    with col2:
        st.metric("Number of Features", len(feature_cols))
    with col3:
        st.metric("Target Variable", target_col)
    
    # Display feature list
    st.write("**Selected Features:**")
    st.write(", ".join(feature_cols))
    
    # Split data: 67% training, 33% testing with random_state=3
    st.subheader("Data Splitting (67% Train, 33% Test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=3
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Training %", f"{len(X_train)/len(X)*100:.1f}%")
    with col3:
        st.metric("Testing Samples", len(X_test))
    with col4:
        st.metric("Testing %", f"{len(X_test)/len(X)*100:.1f}%")
    
    # Standardize features
    st.subheader("Feature Scaling")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.success("✓ Features standardized using StandardScaler")
    
    # Train Linear Regression model
    st.subheader("Model Training")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    st.success("✓ Linear Regression model trained successfully!")
    
    # Display model coefficients
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    st.dataframe(coef_df, use_container_width=True)
    st.write(f"**Intercept:** {model.intercept_:.4f}")
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

# ============================================================================
# STEP 5: TEST - Model Evaluation
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance with comprehensive metrics.
    
    Metrics:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - R² Score
    
    Parameters:
    - model: Trained Linear Regression model
    - X_train, X_test: Training and testing features
    - y_train, y_test: Training and testing targets
    """
    st.header("📊 Step 5: TEST - Model Evaluation")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    st.subheader("Performance Metrics")
    
    # Training metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Testing metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Display metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Metrics")
        st.metric("MSE", f"{train_mse:.4f}")
        st.metric("RMSE", f"{train_rmse:.4f}")
        st.metric("MAE", f"{train_mae:.4f}")
        st.metric("R² Score", f"{train_r2:.4f} ({train_r2*100:.2f}%)")
    
    with col2:
        st.subheader("Testing Metrics")
        st.metric("MSE", f"{test_mse:.4f}")
        st.metric("RMSE", f"{test_rmse:.4f}")
        st.metric("MAE", f"{test_mae:.4f}")
        st.metric("R² Score", f"{test_r2:.4f} ({test_r2*100:.2f}%)")
    
    # Create metrics summary table
    st.subheader("Metrics Summary Table")
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score', 'R² Score (%)'],
        'Training': [f"{train_mse:.6f}", f"{train_rmse:.6f}", f"{train_mae:.6f}", 
                     f"{train_r2:.6f}", f"{train_r2*100:.2f}%"],
        'Testing': [f"{test_mse:.6f}", f"{test_rmse:.6f}", f"{test_mae:.6f}", 
                    f"{test_r2:.6f}", f"{test_r2*100:.2f}%"]
    })
    st.dataframe(metrics_df, use_container_width=True)
    
    # Visualization: Actual vs Predicted
    st.subheader("Actual vs Predicted Values")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training data
    axes[0].scatter(y_train, y_train_pred, alpha=0.5, color='blue')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                 'r--', lw=2)
    axes[0].set_title('Training: Actual vs Predicted', fontweight='bold')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].grid(True, alpha=0.3)
    
    # Testing data
    axes[1].scatter(y_test, y_test_pred, alpha=0.5, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2)
    axes[1].set_title('Testing: Actual vs Predicted', fontweight='bold')
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Visualization: Residuals
    st.subheader("Residual Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    axes[0].hist(train_residuals, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Training Residuals Distribution', fontweight='bold')
    axes[0].set_xlabel('Residuals')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    axes[1].hist(test_residuals, bins=30, color='lightgreen', edgecolor='black')
    axes[1].set_title('Testing Residuals Distribution', fontweight='bold')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return {
        'train_mse': train_mse, 'train_rmse': train_rmse, 'train_mae': train_mae, 'train_r2': train_r2,
        'test_mse': test_mse, 'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
        'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred
    }

# ============================================================================
# STEP 6: DEPLOY - Make Predictions
# ============================================================================

def deploy_predictions(model, scaler, feature_cols):
    """
    Deploy model for making predictions on new customer data.
    
    Parameters:
    - model: Trained Linear Regression model
    - scaler: StandardScaler fitted on training data
    - feature_cols: List of feature column names
    """
    st.header("🚀 Step 6: DEPLOY - Make Predictions")
    
    st.subheader("Predict Customer Spending")
    st.write("Enter customer characteristics to predict yearly spending:")
    
    # Create input fields for each feature
    input_values = {}
    cols = st.columns(len(feature_cols))
    
    for idx, col in enumerate(feature_cols):
        with cols[idx % len(cols)]:
            input_values[col] = st.number_input(
                f"{col}",
                value=0.0,
                step=0.1
            )
    
    # Make prediction
    if st.button("🔮 Predict Spending", key="predict_btn"):
        # Prepare input data
        input_array = np.array([[input_values[col] for col in feature_cols]])
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display prediction
        st.success(f"### Predicted Yearly Amount Spent: ${prediction:.2f}")
        
        # Display input summary
        st.subheader("Input Summary")
        input_df = pd.DataFrame({
            'Feature': feature_cols,
            'Value': [input_values[col] for col in feature_cols]
        })
        st.dataframe(input_df, use_container_width=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main Streamlit application orchestrating all workflow steps.
    """
    # Header
    st.title("🛍️ E-Commerce Customer Analysis Dashboard")
    st.markdown("---")
    st.write("""
    This application performs comprehensive analysis on E-commerce customer data
    using Linear Regression to predict customer spending patterns.
    
    **Workflow:** Analyze → Clean → Visualize → Train → Test → Deploy
    """)
    
    st.markdown("---")
    
    # Step 1: Analyze
    df = load_and_analyze_data()
    
    if df is not None:
        st.markdown("---")
        
        # Step 2: Clean
        df_clean = clean_data(df)
        
        st.markdown("---")
        
        # Step 3: Visualize
        target_col, feature_cols = visualize_data(df_clean)
        
        st.markdown("---")
        
        # Step 4: Train
        model, X_train, X_test, y_train, y_test, scaler, feature_cols = train_model(df_clean, target_col)
        
        st.markdown("---")
        
        # Step 5: Test
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        st.markdown("---")
        
        # Step 6: Deploy
        deploy_predictions(model, scaler, feature_cols)
        
        st.markdown("---")
        
        # Footer
        st.success("""
        ✓ **Project Completed Successfully!**
        
        All workflow steps completed:
        - ✓ Analyze - Dataset exploration
        - ✓ Clean - Data preprocessing
        - ✓ Visualize - Data visualization
        - ✓ Train - Model training (67% training data)
        - ✓ Test - Model evaluation
        - ✓ Deploy - Prediction deployment
        """)

if __name__ == "__main__":
    main()
