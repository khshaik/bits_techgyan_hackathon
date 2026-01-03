Advertising Flask

🚀 Quick Start Guide

Step 1: Install Dependencies
bash
pip install -r requirements_flask.txt

Step 2: Run Flask Application
bash
python newspaper_advertising_flask_analysis.py

Step 3: Open Browser
Navigate to http://localhost:5000

Step 4: Execute Workflow Steps
Click on each step button (1-6) to execute the analysis sequentially

📊 Application Features

Dataset: Advertising.csv (200 rows, 4 columns)
Features: TV, Radio, Newspaper (advertising spend)
Target: Sales (revenue in thousands)

Model: Linear Regression
Training/Testing Split: 67% / 33% (random_state=3)
Feature Scaling: StandardScaler

Performance Metrics:
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R² Score (typically 0.89-0.91)

Visualizations:
Distribution plots
Correlation heatmaps
Feature distributions
Scatter plots (feature vs target)
Actual vs Predicted plots
Residual distributions


📁 Project Structure
FUNNYAPP/
├── advertising_flask_app.py              # Main Flask app
├── requirements_flask.txt                # Dependencies
├── templates/
│   └── index.html                        # Web interface
├── advertising_analysis.ipynb            # Jupyter notebook
└── ADVERTISING_FLASK_DEPLOYMENT.md      # Deployment guide