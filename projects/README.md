# BITS Hackathon - Machine Learning Projects Suite

## 📋 Project Overview

The BITS Hackathon project is a comprehensive machine learning suite containing three independent data science projects demonstrating end-to-end ML workflows. Each project showcases different algorithms, datasets, and deployment strategies suitable for students and professionals learning applied machine learning.

### 🎯 Project Objectives

- **Educational Value**: Demonstrate complete ML pipelines from data exploration to deployment
- **Practical Application**: Real-world datasets with business significance
- **Multiple Frameworks**: Jupyter Notebooks, Flask, and Streamlit implementations
- **Scalability**: Modular architecture for easy extension and customization

---

## 📁 Project Structure

```
BITS_TECHGYAN_HACKATHON/
│
├── README.md                          # Overall project overview
├── SETUP.md                           # Common setup instructions
│
├── advertising/                       # Advertising Spend Analysis (Flask)
│   │
│   ├── README.md                      # Advertising project overview
│   ├── SETUP.md                       # Advertising-specific setup
│   │
│   ├── algorithm/
│   │   └── ALGORITHM_LINEAR_REGRESSION.md
│   │
│   ├── code/
│   │   ├── newspaper_advertising_flask_analysis.ipynb
│   │   └── newspaper_advertising_flask_analysis.py
│   │
│   ├── notes/
│   │   └── ADVERTISING_FLASK_DEPLOYMENT.md
│   │
│   ├── output/
│   │   └── flask_analysis.pdf
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── requirements_flask.txt
│
├── ecommerce/                         # E-commerce Analysis
│   │
│   ├── README.md
│   ├── SETUP.md
│   │
│   ├── algorithm/
│   │   └── ALGORITHM_LINEAR_REGRESSION_ECOMMERCE.md
│   │
│   ├── code/
│   │   ├── ecommerce_customer_analysis.ipynb
│   │   └── ecommerce_customer_analysis.py
│   │
│   ├── notes/
│   │   └── ARCHITECTURE.md
│   │
│   ├── output/
│   │   └── ecommerce_analysis.pdf
│   │
│   └── setup/
│       └── SETUP.md
│
├── titanic/                           # Titanic Survival Prediction
│   │
│   ├── README.md
│   ├── SETUP.md
│   │
│   ├── algorithm/
│   │   └── ALGORITHM_DECISION_TREE.md
│   │
│   ├── code/
│   │   ├── titanic_data_analysis.ipynb
│   │   └── titanic_data_analysis.py
│   │
│   ├── notes/
│   │   └── TITANIC_JUPYTER_DEPLOYMENT.md
│   │
│   └── setup/
│       └── SETUP.md
│
└── notes/
    └── ARCHITECTURE.md                # Overall system architecture
```

---

## 🚀 Quick Start

### For Beginners
1. Read this README for project overview
2. Choose a sub-project that interests you
3. Follow the specific project's SETUP.md
4. Run the application (Jupyter, Flask, or Streamlit)
5. Explore the code and modify as needed

### For Experienced Developers
1. Review ARCHITECTURE.md for system design
2. Check individual project READMEs for specifics
3. Install dependencies: `pip install -r requirements.txt`
4. Run applications directly
5. Extend or customize as needed

---

## 📊 Sub-Projects Overview

### 1. **Advertising Analysis** (Flask Web Application)
**Domain**: Marketing Analytics & Sales Forecasting

- **Dataset**: Advertising spend across TV, Radio, Newspaper channels
- **Algorithm**: Linear Regression
- **Deployment**: Flask web application with interactive UI
- **Key Metrics**: MSE, RMSE, MAE, R² Score
- **Use Case**: Predict sales based on advertising budget allocation

**Documentation**:
- `README.md` - Project overview and features
- `SETUP.md` - Installation and setup instructions
- `ALGORITHM_LINEAR_REGRESSION.md` - Algorithm theory, mathematics, and why it was chosen
- `ADVERTISING_FLASK_DEPLOYMENT.md` - Deployment guide

**Quick Start**:
```bash
cd advertising
pip install -r requirements_flask.txt
python newspaper_advertising_flask_analysis.py
# Open http://localhost:5000
```

---

### 2. **E-commerce Customer Analysis** (Streamlit Dashboard)
**Domain**: Customer Analytics & Spending Prediction

- **Dataset**: E-commerce customer characteristics and spending patterns
- **Algorithm**: Linear Regression
- **Deployment**: Streamlit interactive dashboard
- **Key Metrics**: MSE, RMSE, MAE, R² Score
- **Use Case**: Predict customer spending based on demographics

**Documentation**:
- `README.md` - Project overview and features
- `SETUP.md` - Installation and setup instructions
- `ALGORITHM_LINEAR_REGRESSION_ECOMMERCE.md` - Algorithm theory, mathematics, and why it was chosen

**Quick Start**:
```bash
cd ecommerce
pip install -r requirements_streamlit.txt
streamlit run ecommerce_customer_streamlit_analysis.py
# Opens at http://localhost:8501
```

---

### 3. **Titanic Survival Prediction** (Jupyter Notebook & Python Script)
**Domain**: Classification & Survival Analysis

- **Dataset**: Titanic passenger data with survival outcomes
- **Algorithm**: Decision Tree Classifier
- **Deployment**: Jupyter Notebook and standalone Python script
- **Key Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Use Case**: Predict passenger survival based on demographics and ticket information

**Documentation**:
- `README.md` - Project overview and features
- `SETUP.md` - Installation and setup instructions
- `ALGORITHM_DECISION_TREE.md` - Algorithm theory, mathematics, and why it was chosen
- `TITANIC_JUPYTER_DEPLOYMENT.md` - Deployment guide

**Quick Start**:
```bash
cd titanic
pip install -r requirements.txt
# Option 1: Jupyter Notebook
jupyter notebook BITS_AIML_Titanic_Jan3rd2026.ipynb

# Option 2: Python Script
python BITS_AIML_Titanic_Jan3rd2026.py
```

---

## 🏗️ Architectural Overview

### Technology Stack

| Component | Technology | Projects |
|-----------|-----------|----------|
| **Data Processing** | Pandas, NumPy | All |
| **ML Algorithms** | Scikit-learn | All |
| **Visualization** | Matplotlib, Seaborn | All |
| **Web Framework** | Flask | Advertising |
| **Dashboard** | Streamlit | E-commerce |
| **Notebooks** | Jupyter | All |
| **Data Source** | GitHub (CSV) | All |

### Workflow Architecture

Each project follows a standardized 6-step ML pipeline:

```
1. ANALYZE
   ├─ Load dataset
   ├─ Explore structure
   └─ Identify patterns

2. CLEAN
   ├─ Handle missing values
   ├─ Remove duplicates
   └─ Encode categorical variables

3. VISUALIZE
   ├─ Distribution analysis
   ├─ Correlation heatmaps
   └─ Feature relationships

4. TRAIN
   ├─ Feature selection
   ├─ Data splitting (67/33)
   ├─ Feature scaling
   └─ Model training

5. TEST
   ├─ Make predictions
   ├─ Calculate metrics
   └─ Analyze performance

6. DEPLOY
   ├─ Create predictions
   ├─ Interactive interface
   └─ Real-world application
```

---

## 📚 Learning Path

### For Students

**Beginner Level**:
1. Start with Titanic project (Decision Trees are intuitive)
2. Run Jupyter notebook to see step-by-step execution
3. Modify hyperparameters and observe results
4. Read code comments and documentation

**Intermediate Level**:
1. Explore Advertising project (Flask deployment)
2. Understand web application structure
3. Learn how to serve ML models in production
4. Modify HTML templates and styling

**Advanced Level**:
1. Study E-commerce project (Streamlit dashboard)
2. Build interactive data applications
3. Implement custom visualizations
4. Extend with additional features

### For Professionals

1. **Code Review**: Examine best practices in ML pipeline implementation
2. **Architecture**: Study deployment patterns (Flask, Streamlit, Jupyter)
3. **Scalability**: Understand how to extend projects
4. **Integration**: Learn data pipeline and model serving patterns

---

## 🔧 Technical Highlights

### Data Pipeline
- Automated data loading from GitHub
- Robust missing value handling
- Categorical variable encoding
- Feature scaling and normalization

### Model Training
- Train/test split with fixed random state (reproducibility)
- Feature standardization using StandardScaler
- Comprehensive performance metrics
- Cross-validation ready architecture

### Deployment Options
- **Jupyter Notebooks**: Interactive exploration and learning
- **Flask**: Traditional web application with HTML/CSS
- **Streamlit**: Modern dashboard with minimal code
- **Python Scripts**: Standalone execution for batch processing

---

## 📖 Documentation Structure

Each sub-project contains:

1. **README.md**: Project-specific overview, features, and usage
2. **SETUP.md**: Step-by-step installation and configuration
3. **Source Code**: Well-commented Python files
4. **Requirements**: Dependency specifications
5. **Templates/Assets**: Web UI components (where applicable)

---

## 🎓 Key Concepts Covered

### Data Science
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Selection and Training
- Performance Evaluation
- Hyperparameter Tuning

### Machine Learning
- Linear Regression (Advertising, E-commerce)
- Decision Tree Classification (Titanic)
- Cross-validation
- Overfitting/Underfitting
- Regularization concepts

### Software Engineering
- Project structure and organization
- Code documentation and comments
- Error handling and validation
- Web application development
- Dashboard creation
- Version control ready

---

## 🔗 External Resources

### Datasets
- **Advertising**: [GitHub Datasets](https://github.com/erkansirin78/datasets/blob/master/Advertising.csv)
- **E-commerce**: [GitHub Datasets](https://github.com/erkansirin78/datasets)
- **Titanic**: [GitHub Datasets](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)

### Libraries Documentation
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Streamlit](https://streamlit.io/)
- [Jupyter](https://jupyter.org/)

---

## 💡 Tips for Success

### Setup
- Use virtual environments to avoid dependency conflicts
- Install exact versions specified in requirements files
- Verify internet connection for GitHub data downloads

### Learning
- Run projects step-by-step, don't skip sections
- Modify code and observe changes
- Experiment with hyperparameters
- Read error messages carefully

### Development
- Keep code modular and reusable
- Add comments for complex logic
- Test changes incrementally
- Use version control (Git)

---

## 🤝 Contributing

To extend or improve projects:

1. Create a new branch for your changes
2. Follow existing code style and structure
3. Add documentation for new features
4. Test thoroughly before committing
5. Update relevant README files

---

## 📝 License

These projects are created for educational purposes as part of the BITS Hackathon initiative.

---

## 🆘 Troubleshooting

### Common Issues

**Import Errors**:
- Ensure virtual environment is activated
- Verify all dependencies are installed: `pip list`
- Check Python version compatibility (3.7+)

**Data Loading Errors**:
- Check internet connection
- Verify GitHub URLs are accessible
- Ensure pandas is properly installed

**Port Already in Use**:
- Flask: Change port in app.run() or use `lsof -i :5000`
- Streamlit: Use `streamlit run --server.port 8502`

**Matplotlib Errors**:
- Ensure matplotlib backend is set correctly
- For headless systems, use 'Agg' backend

---

## 📞 Support

For questions or issues:
1. Check project-specific README.md
2. Review SETUP.md for installation help
3. Examine code comments and docstrings
4. Refer to library documentation

---

**Last Updated**: January 2026
**Version**: 1.0
**Status**: Production Ready
