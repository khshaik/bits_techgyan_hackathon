Ecommerce Streamlit

requirements_streamlit.txt
All required dependencies:
streamlit==1.28.1
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

3. ECOMMERCE_STREAMLIT_README.md
Comprehensive documentation including:
Project overview and workflow steps
Installation instructions
How to use the application
Dataset information
Model details and performance metrics
Troubleshooting guide
Code structure and learning resources

🚀 How to Run the Application

Step 1: Install Dependencies
bash
pip install -r requirements_streamlit.txt

Step 2: Run Streamlit App
bash
streamlit run ecommerce_customer_streamlit_analysis.py

The application will open at http://localhost:8501

📊 Application Features
Data Analysis
Loads E-commerce customer dataset from GitHub
Displays dataset dimensions, columns, and statistics
Identifies and handles missing values
Removes duplicate records
Visualizations
Target variable distribution (histogram & box plot)
Correlation heatmap
Feature distributions
Scatter plots (features vs target)
Actual vs Predicted plots
Residual distributions
Model Training
Linear Regression algorithm
Feature standardization (StandardScaler)
67% training / 33% testing split
Fixed random_state=3 for reproducibility
Performance Metrics
MSE, RMSE, MAE - Regression error metrics
R² Score - Variance explained (0-1 scale)
Training vs Testing comparison
Residual analysis
Interactive Predictions
Input customer characteristics
Real-time spending prediction
Display prediction results
Show input summary