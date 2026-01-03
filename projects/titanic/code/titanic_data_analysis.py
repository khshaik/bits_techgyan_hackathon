# Import required libraries for data analysis and machine learning
import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # Statistical data visualization
from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.tree import DecisionTreeClassifier  # Decision Tree algorithm
from sklearn.preprocessing import LabelEncoder  # Encode categorical variables
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, confusion_matrix,
                             f1_score, classification_report)  # Performance metrics
import warnings
warnings.filterwarnings('ignore')  # Suppress warning messages

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load Titanic dataset from GitHub
# Source: https://github.com/datasciencedojo/datasets/blob/master/titanic.csv
titanic_df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv")

# Display first few rows to understand data structure
print("=" * 80)
print("TITANIC DATASET - FIRST 5 ROWS")
print("=" * 80)
print(titanic_df.head())
print("\n")

# Display dataset shape (rows, columns)
print("=" * 80)
print("DATASET DIMENSIONS")
print("=" * 80)
print(f"Total Rows: {titanic_df.shape[0]}")
print(f"Total Columns: {titanic_df.shape[1]}")
print("\n")

# Display column names and data types
print("=" * 80)
print("COLUMN INFORMATION")
print("=" * 80)
print(titanic_df.info())
print("\n")

# Display basic statistics
print("=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)
print(titanic_df.describe())

# Analyze missing values in the dataset
print("=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)

# Count missing values per column
missing_values = titanic_df.isnull().sum()
missing_percentage = (titanic_df.isnull().sum() / len(titanic_df)) * 100

# Create a summary dataframe
missing_summary = pd.DataFrame({
    'Column': titanic_df.columns,
    'Missing_Count': missing_values.values,
    'Missing_Percentage': missing_percentage.values
})

print(missing_summary)
print("\n")

# Display columns with missing values
print("=" * 80)
print("COLUMNS WITH MISSING DATA")
print("=" * 80)
missing_cols = missing_summary[missing_summary['Missing_Count'] > 0]
print(missing_cols)
print("\n")

# Step 1: Select relevant features for the model
# We'll use: Survived (target), Pclass, Sex, Age, SibSp, Parch, Embarked
print("=" * 80)
print("FEATURE SELECTION")
print("=" * 80)
print("Selected features for Decision Tree model:")
print("- Survived (Target variable)")
print("- Pclass (Passenger class: 1, 2, or 3)")
print("- Sex (Male or Female)")
print("- Age (Passenger age in years)")
print("- SibSp (Number of siblings/spouses aboard)")
print("- Parch (Number of parents/children aboard)")
print("- Embarked (Port of embarkation: C, Q, or S)")
print("\n")

# Create a new dataframe with selected features
titanic_clean = titanic_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].copy()

print("Selected dataset shape:", titanic_clean.shape)
print("\nFirst 5 rows of selected features:")
print(titanic_clean.head())
print("\n")

# Step 2: Handle missing Age values
# Strategy: Fill missing Age values based on Passenger Class average
# Rationale: Different passenger classes had different age distributions
print("=" * 80)
print("HANDLING MISSING AGE VALUES")
print("=" * 80)

# Calculate mean age for each passenger class
age_by_class = titanic_clean.groupby('Pclass')['Age'].mean()
print("Mean Age by Passenger Class:")
print(age_by_class)
print("\n")

# Define function to fill missing Age values based on Pclass
def fill_missing_age(row):
    """
    Fill missing Age values with class-specific mean age

    Parameters:
    - row: DataFrame row containing Age and Pclass

    Returns:
    - Age value (original if not missing, class mean if missing)
    """
    if pd.isnull(row['Age']):
        # Return class-specific mean age
        return age_by_class[row['Pclass']]
    else:
        # Return original age if not missing
        return row['Age']

# Apply the function to fill missing Age values
titanic_clean['Age'] = titanic_clean.apply(fill_missing_age, axis=1)

print("Missing Age values after filling:", titanic_clean['Age'].isnull().sum())
print("\n")

# Step 3: Handle missing Embarked values
# Strategy: Fill with most common port (mode)
print("=" * 80)
print("HANDLING MISSING EMBARKED VALUES")
print("=" * 80)

# Find the most common embarkation port
embarked_mode = titanic_clean['Embarked'].mode()[0]
print(f"Most common embarkation port: {embarked_mode}")

# Fill missing Embarked values with the mode
titanic_clean['Embarked'].fillna(embarked_mode, inplace=True)

print(f"Missing Embarked values after filling: {titanic_clean['Embarked'].isnull().sum()}")
print("\n")

# Verify no missing values remain
print("=" * 80)
print("MISSING VALUES VERIFICATION")
print("=" * 80)
print(titanic_clean.isnull().sum())
print("\nAll missing values have been successfully handled!")
print("\n")

# Step 4: Encode categorical variables
# Convert categorical features (Sex, Embarked) to numerical values
print("=" * 80)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 80)

# Initialize LabelEncoder for converting categorical to numerical
label_encoder = LabelEncoder()

# Encode 'Sex' column (Male -> 1, Female -> 0)
print("Encoding 'Sex' column:")
print(f"Unique values before encoding: {titanic_clean['Sex'].unique()}")
titanic_clean['Sex'] = label_encoder.fit_transform(titanic_clean['Sex'])
print(f"Encoding: Female=0, Male=1")
print(f"Unique values after encoding: {titanic_clean['Sex'].unique()}")
print("\n")

# Encode 'Embarked' column (C, Q, S -> numerical)
print("Encoding 'Embarked' column:")
print(f"Unique values before encoding: {titanic_clean['Embarked'].unique()}")
titanic_clean['Embarked'] = label_encoder.fit_transform(titanic_clean['Embarked'])
print(f"Unique values after encoding: {titanic_clean['Embarked'].unique()}")
print("\n")

# Display cleaned dataset
print("=" * 80)
print("CLEANED DATASET - FIRST 10 ROWS")
print("=" * 80)
print(titanic_clean.head(10))
print("\n")

print("=" * 80)
print("CLEANED DATASET INFO")
print("=" * 80)
print(titanic_clean.info())
print("\nData cleaning completed successfully!")

# Visualization 1: Survival Distribution
# Shows the count of passengers who survived vs did not survive
print("=" * 80)
print("VISUALIZATION 1: SURVIVAL DISTRIBUTION")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot for survival
sns.countplot(data=titanic_clean, x='Survived', ax=axes[0], palette='Set2')
axes[0].set_title('Survival Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Survived (0=No, 1=Yes)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)

# Add value labels on bars
for container in axes[0].containers:
    axes[0].bar_label(container)

# Pie chart for survival percentage
survival_counts = titanic_clean['Survived'].value_counts()
axes[1].pie(survival_counts, labels=['Did Not Survive', 'Survived'],
            autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90)
axes[1].set_title('Survival Percentage', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"Total Passengers: {len(titanic_clean)}")
print(f"Survived: {survival_counts[1]} ({survival_counts[1]/len(titanic_clean)*100:.2f}%)")
print(f"Did Not Survive: {survival_counts[0]} ({survival_counts[0]/len(titanic_clean)*100:.2f}%)")
print("\n")

# Visualization 2: Survival by Passenger Class
# Shows how survival rate varies across different passenger classes
print("=" * 80)
print("VISUALIZATION 2: SURVIVAL BY PASSENGER CLASS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot: Survival by Pclass
sns.countplot(data=titanic_clean, x='Pclass', hue='Survived', ax=axes[0], palette='Set1')
axes[0].set_title('Survival Count by Passenger Class', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Passenger Class', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].legend(['Did Not Survive', 'Survived'], loc='upper right')

# Survival rate by Pclass
survival_by_class = titanic_clean.groupby('Pclass')['Survived'].mean() * 100
survival_by_class.plot(kind='bar', ax=axes[1], color=['#ff9999', '#66b3ff', '#99ff99'])
axes[1].set_title('Survival Rate by Passenger Class', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Passenger Class', fontsize=11)
axes[1].set_ylabel('Survival Rate (%)', fontsize=11)
axes[1].set_xticklabels(['Class 1', 'Class 2', 'Class 3'], rotation=0)

plt.tight_layout()
plt.show()

print("Survival Rate by Passenger Class:")
for pclass, rate in survival_by_class.items():
    print(f"  Class {pclass}: {rate:.2f}%")
print("\n")

# Visualization 3: Survival by Gender
# Shows gender-based survival patterns (women and children first policy)
print("=" * 80)
print("VISUALIZATION 3: SURVIVAL BY GENDER")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot: Survival by Sex
sns.countplot(data=titanic_clean, x='Sex', hue='Survived', ax=axes[0], palette='Set2')
axes[0].set_title('Survival Count by Gender', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Gender (0=Female, 1=Male)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].legend(['Did Not Survive', 'Survived'], loc='upper right')

# Survival rate by Sex
survival_by_sex = titanic_clean.groupby('Sex')['Survived'].mean() * 100
survival_by_sex.plot(kind='bar', ax=axes[1], color=['#ffcc99', '#99ccff'])
axes[1].set_title('Survival Rate by Gender', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Gender', fontsize=11)
axes[1].set_ylabel('Survival Rate (%)', fontsize=11)
axes[1].set_xticklabels(['Female', 'Male'], rotation=0)

plt.tight_layout()
plt.show()

print("Survival Rate by Gender:")
print(f"  Female (0): {survival_by_sex[0]:.2f}%")
print(f"  Male (1): {survival_by_sex[1]:.2f}%")
print("\n")

# Visualization 4: Age Distribution and Survival
# Shows how age relates to survival chances
print("=" * 80)
print("VISUALIZATION 4: AGE DISTRIBUTION AND SURVIVAL")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram: Age distribution by survival
axes[0].hist([titanic_clean[titanic_clean['Survived']==0]['Age'],
              titanic_clean[titanic_clean['Survived']==1]['Age']],
             label=['Did Not Survive', 'Survived'], bins=20, color=['#ff9999', '#66b3ff'])
axes[0].set_title('Age Distribution by Survival Status', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Age (years)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].legend()

# Box plot: Age by survival
sns.boxplot(data=titanic_clean, x='Survived', y='Age', ax=axes[1], palette='Set2')
axes[1].set_title('Age Distribution by Survival (Box Plot)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Survived (0=No, 1=Yes)', fontsize=11)
axes[1].set_ylabel('Age (years)', fontsize=11)

plt.tight_layout()
plt.show()

print("Age Statistics by Survival:")
print(titanic_clean.groupby('Survived')['Age'].describe())
print("\n")

# Prepare features (X) and target variable (y)
print("=" * 80)
print("PREPARING DATA FOR MODEL TRAINING")
print("=" * 80)

# X contains all features except 'Survived'
X = titanic_clean.drop('Survived', axis=1)

# y contains only the target variable 'Survived'
y = titanic_clean['Survived']

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("\nFeature columns:")
print(X.columns.tolist())
print("\nTarget variable distribution:")
print(y.value_counts())
print("\n")

# Display first few rows of features
print("=" * 80)
print("FIRST 5 ROWS OF FEATURES (X)")
print("=" * 80)
print(X.head())
print("\n")

# Split data into training and testing sets
# Training: 67% (0.67), Testing: 33% (0.33), Random State: 3
print("=" * 80)
print("SPLITTING DATA INTO TRAIN AND TEST SETS")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,  # 33% of data for testing
    random_state=3   # Fixed seed for reproducibility
)

print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
print(f"Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")
print("\n")

# Display training data distribution
print("=" * 80)
print("TRAINING DATA DISTRIBUTION")
print("=" * 80)
print(f"Training set shape: {X_train.shape}")
print(f"Training target distribution:")
print(y_train.value_counts())
print("\n")

# Display testing data distribution
print("=" * 80)
print("TESTING DATA DISTRIBUTION")
print("=" * 80)
print(f"Testing set shape: {X_test.shape}")
print(f"Testing target distribution:")
print(y_test.value_counts())
print("\n")

# Initialize and train Decision Tree Classifier
print("=" * 80)
print("TRAINING DECISION TREE CLASSIFIER")
print("=" * 80)

# Create Decision Tree Classifier object
# Parameters:
# - random_state=3: For reproducibility
# - max_depth=5: Limit tree depth to prevent overfitting
# - min_samples_split=10: Minimum samples required to split a node
dt_classifier = DecisionTreeClassifier(
    random_state=3,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)

print("Decision Tree Classifier initialized with parameters:")
print(f"  - random_state: 3")
print(f"  - max_depth: 5")
print(f"  - min_samples_split: 10")
print(f"  - min_samples_leaf: 5")
print("\n")

# Train the model on training data
print("Training model on training data...")
dt_classifier.fit(X_train, y_train)

print("✓ Model training completed successfully!")
print("\n")

# Display model information
print("=" * 80)
print("MODEL INFORMATION")
print("=" * 80)
print(f"Model type: {type(dt_classifier).__name__}")
print(f"Number of features: {dt_classifier.n_features_in_}")
print(f"Feature names: {X_train.columns.tolist()}")
print(f"Tree depth: {dt_classifier.get_depth()}")
print(f"Number of leaves: {dt_classifier.get_n_leaves()}")
print("\n")

# Make predictions on test data
print("=" * 80)
print("MAKING PREDICTIONS ON TEST DATA")
print("=" * 80)

# Predict survival for test set
y_pred = dt_classifier.predict(X_test)

print(f"Predictions made for {len(y_pred)} test samples")
print(f"\nPrediction distribution:")
print(f"  Did Not Survive (0): {sum(y_pred == 0)}")
print(f"  Survived (1): {sum(y_pred == 1)}")
print("\n")

# Display first 10 predictions vs actual values
print("=" * 80)
print("FIRST 10 PREDICTIONS VS ACTUAL VALUES")
print("=" * 80)
comparison_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Match': y_test.values[:10] == y_pred[:10]
})
print(comparison_df)
print("\n")

# Calculate performance metrics
print("=" * 80)
print("PERFORMANCE METRICS CALCULATION")
print("=" * 80)

# 1. Mean Squared Error (MSE)
# Measures average squared difference between predicted and actual values
# Lower values indicate better fit
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.6f}")
print("  - Interpretation: Average squared error in predictions")
print("  - Lower is better")
print("\n")

# 2. Root Mean Squared Error (RMSE)
# Square root of MSE, in same units as target variable
# Easier to interpret than MSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print("  - Interpretation: Average error magnitude in prediction")
print("  - Lower is better")
print("\n")

# 3. Mean Absolute Error (MAE)
# Average absolute difference between predicted and actual values
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print("  - Interpretation: Average absolute error")
print("  - Lower is better")
print("\n")

# 4. R² Score (Coefficient of Determination)
# Measures proportion of variance explained by the model
# Range: 0 to 1 (1 is perfect prediction)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.6f}")
print(f"R² Score (%): {r2*100:.2f}%")
print("  - Interpretation: Proportion of variance explained")
print("  - Higher is better (max 1.0 or 100%)")
print("\n")

# 5. Accuracy Score
# Percentage of correct predictions out of total predictions
# Most intuitive metric for classification problems
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.6f}")
print(f"Accuracy Score (%): {accuracy*100:.2f}%")
print("  - Interpretation: Percentage of correct predictions")
print("  - Higher is better (max 1.0 or 100%)")
print("\n")

# 6. Confusion Matrix
# Shows True Positives, True Negatives, False Positives, False Negatives
cm = confusion_matrix(y_test, y_pred)
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
print(cm)
print("\nConfusion Matrix Breakdown:")
tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives (TN):  {tn} - Correctly predicted 'Did Not Survive'")
print(f"  False Positives (FP): {fp} - Incorrectly predicted 'Survived'")
print(f"  False Negatives (FN): {fn} - Incorrectly predicted 'Did Not Survive'")
print(f"  True Positives (TP):  {tp} - Correctly predicted 'Survived'")
print("\n")

# 7. F1 Score
# Harmonic mean of Precision and Recall
# Useful when dealing with imbalanced datasets
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.6f}")
print(f"F1 Score (%): {f1*100:.2f}%")
print("  - Interpretation: Balance between precision and recall")
print("  - Higher is better (max 1.0 or 100%)")
print("\n")

# 8. Classification Report
# Detailed breakdown of Precision, Recall, F1-Score for each class
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test, y_pred,
                          target_names=['Did Not Survive', 'Survived']))

print("\nClassification Report Interpretation:")
print("  Precision: Of predicted positives, how many were actually positive")
print("  Recall: Of actual positives, how many were correctly predicted")
print("  F1-Score: Harmonic mean of precision and recall")
print("  Support: Number of samples in each class")
print("\n")

# Visualize Confusion Matrix
print("=" * 80)
print("CONFUSION MATRIX VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap of confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix - Heatmap', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=11)
axes[0].set_xlabel('Predicted', fontsize=11)

# Calculate metrics from confusion matrix
sensitivity = tp / (tp + fn)  # True Positive Rate / Recall
specificity = tn / (tn + fp)  # True Negative Rate
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

# Bar plot of metrics
metrics = ['Accuracy', 'Precision', 'Recall\n(Sensitivity)', 'Specificity', 'F1-Score']
values = [accuracy, precision, sensitivity, specificity, f1]
colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999', '#cc99ff']

axes[1].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Score', fontsize=11)
axes[1].set_ylim([0, 1.1])
axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')

# Add value labels on bars
for i, v in enumerate(values):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

axes[1].legend()
plt.tight_layout()
plt.show()

print(f"Sensitivity (Recall): {sensitivity:.6f}")
print(f"Specificity: {specificity:.6f}")
print(f"Precision: {precision:.6f}")
print("\n")
