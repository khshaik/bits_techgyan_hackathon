# Linear Regression Algorithm - Advertising Project

## 📚 Overview

Linear Regression is a fundamental supervised learning algorithm used to model the linear relationship between input features (independent variables) and a continuous output (dependent variable). In the Advertising project, it predicts sales revenue based on advertising spending across TV, Radio, and Newspaper channels.

---

## 🎯 Algorithm Significance

### Why Linear Regression Matters

1. **Interpretability**: Coefficients directly show the impact of each feature on the target
2. **Simplicity**: Easy to understand and implement
3. **Efficiency**: Fast training and prediction
4. **Baseline Model**: Provides baseline performance for comparison
5. **Real-world Application**: Widely used in business analytics and forecasting
6. **Mathematical Foundation**: Foundation for more complex algorithms

### Key Characteristics

- **Supervised Learning**: Requires labeled training data
- **Regression Task**: Predicts continuous numerical values
- **Linear Relationship**: Assumes linear relationship between features and target
- **Parametric Model**: Learns fixed number of parameters
- **Deterministic**: Same input always produces same output

---

## 🔧 How Linear Regression Works

### Conceptual Understanding

Linear Regression finds the best-fitting straight line (or hyperplane in multiple dimensions) through the data points that minimizes prediction errors.

```
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper) + ε

Where:
- Sales: Predicted sales revenue (target)
- β₀: Intercept (baseline sales when all features are 0)
- β₁, β₂, β₃: Coefficients (weights) for each feature
- TV, Radio, Newspaper: Input features (advertising spend)
- ε: Error term (residual)
```

### Step-by-Step Process

#### Step 1: Initialize Parameters
```
β₀, β₁, β₂, β₃ = random initial values or zeros
```

#### Step 2: Make Predictions
```
ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃

Where:
- ŷ: Predicted value
- x₁, x₂, x₃: Feature values
```

#### Step 3: Calculate Error (Loss)
```
Error = y - ŷ (Residual for each sample)
```

#### Step 4: Optimize Parameters
Using Ordinary Least Squares (OLS) method to minimize sum of squared errors:

```
Minimize: SSE = Σ(yᵢ - ŷᵢ)²

Solution: β = (XᵀX)⁻¹Xᵀy

Where:
- X: Feature matrix (including intercept column)
- y: Target vector
- β: Coefficient vector
```

#### Step 5: Evaluate Performance
Calculate metrics like MSE, RMSE, MAE, R²

---

## 📐 Mathematical Formulas

### 1. Linear Regression Model

**Simple Linear Regression (Single Feature)**:
```
ŷ = β₀ + β₁x
```

**Multiple Linear Regression (Multiple Features)**:
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

In vector form:
ŷ = Xβ

Where:
X = [1  x₁₁  x₁₂  ...  x₁ₙ]
    [1  x₂₁  x₂₂  ...  x₂ₙ]
    [⋮   ⋮    ⋮    ⋱   ⋮  ]
    [1  xₘ₁  xₘ₂  ...  xₘₙ]

β = [β₀]
    [β₁]
    [β₂]
    [⋮ ]
    [βₙ]
```

### 2. Cost Function (Loss Function)

**Mean Squared Error (MSE)**:
```
MSE = (1/m) Σᵢ₌₁ᵐ (yᵢ - ŷᵢ)²

Where:
- m: Number of samples
- yᵢ: Actual value
- ŷᵢ: Predicted value
```

**Sum of Squared Errors (SSE)**:
```
SSE = Σᵢ₌₁ᵐ (yᵢ - ŷᵢ)²
```

### 3. Parameter Estimation (Ordinary Least Squares)

**Normal Equation**:
```
β = (XᵀX)⁻¹Xᵀy

Where:
- Xᵀ: Transpose of X
- (XᵀX)⁻¹: Inverse of (XᵀX)
- y: Target vector
```

**Derivation**:
```
Minimize: L(β) = (y - Xβ)ᵀ(y - Xβ)

Taking derivative with respect to β:
∂L/∂β = -2Xᵀ(y - Xβ) = 0

Solving for β:
XᵀXβ = Xᵀy
β = (XᵀX)⁻¹Xᵀy
```

### 4. Predictions

**Single Sample Prediction**:
```
ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃
```

**Multiple Sample Predictions**:
```
ŷ = Xβ
```

### 5. Residuals

**Residual for Each Sample**:
```
eᵢ = yᵢ - ŷᵢ
```

**Residual Sum of Squares**:
```
RSS = Σᵢ₌₁ᵐ eᵢ² = Σᵢ₌₁ᵐ (yᵢ - ŷᵢ)²
```

### 6. Performance Metrics

**Mean Squared Error (MSE)**:
```
MSE = (1/m) Σᵢ₌₁ᵐ (yᵢ - ŷᵢ)²
```

**Root Mean Squared Error (RMSE)**:
```
RMSE = √MSE = √[(1/m) Σᵢ₌₁ᵐ (yᵢ - ŷᵢ)²]
```

**Mean Absolute Error (MAE)**:
```
MAE = (1/m) Σᵢ₌₁ᵐ |yᵢ - ŷᵢ|
```

**R² Score (Coefficient of Determination)**:
```
R² = 1 - (SSres/SStot)

Where:
- SSres = Σᵢ₌₁ᵐ (yᵢ - ŷᵢ)²  (Residual Sum of Squares)
- SStot = Σᵢ₌₁ᵐ (yᵢ - ȳ)²   (Total Sum of Squares)
- ȳ = (1/m) Σᵢ₌₁ᵐ yᵢ        (Mean of actual values)

Interpretation:
- R² = 1: Perfect fit
- R² = 0: Model explains no variance
- R² < 0: Model worse than horizontal line
```

**Adjusted R²** (accounts for number of features):
```
Adjusted R² = 1 - [(1 - R²)(m - 1)/(m - p - 1)]

Where:
- m: Number of samples
- p: Number of features
```

### 7. Feature Importance

**Standardized Coefficients** (for feature importance):
```
β_standardized = β × (σₓ/σᵧ)

Where:
- σₓ: Standard deviation of feature x
- σᵧ: Standard deviation of target y
```

---

## 🎓 Learning Perspective

### Assumptions of Linear Regression

1. **Linearity**: Relationship between features and target is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No Multicollinearity**: Features are not highly correlated

### Advantages

✓ Simple and interpretable
✓ Fast training and prediction
✓ Works well with linear relationships
✓ Provides confidence intervals
✓ Computationally efficient
✓ Good baseline model
✓ Coefficients show feature impact

### Disadvantages

✗ Assumes linear relationships
✗ Sensitive to outliers
✗ Assumes constant variance
✗ Cannot capture non-linear patterns
✗ Requires feature scaling for some algorithms
✗ Assumes independence of features
✗ May underfit complex relationships

---

## 🎯 Why Linear Regression for Advertising Dataset

### Dataset Characteristics

```
Dataset: Advertising
- Samples: 200
- Features: 3 (TV, Radio, Newspaper)
- Target: Sales (continuous)
- Feature Type: All numerical
- Relationship: Appears linear
```

### Reasons for Selection

#### 1. **Linear Relationship in Data**
```
Correlation Analysis:
TV ↔ Sales:        0.78 (Strong positive)
Radio ↔ Sales:     0.58 (Moderate positive)
Newspaper ↔ Sales: 0.23 (Weak positive)

→ Clear linear relationships visible
```

#### 2. **Problem Type**
- **Regression Task**: Predicting continuous sales values
- **Not Classification**: Not predicting categories
- **Linear Regression** is ideal for continuous prediction

#### 3. **Feature Characteristics**
- All features are numerical (no encoding needed)
- Features are continuous (not categorical)
- Features are on similar scales
- No complex interactions apparent

#### 4. **Dataset Size**
- 200 samples is sufficient for linear regression
- Not too small (would overfit easily)
- Not too large (computational efficiency)
- Good for learning purposes

#### 5. **Interpretability Requirement**
- Business stakeholders want to understand feature impact
- Linear Regression provides clear coefficient interpretation
- Easy to explain: "Each $1000 TV spend increases sales by $X"

#### 6. **Performance Expectations**
- Linear Regression achieves R² ≈ 0.87-0.91
- Strong predictive power on this dataset
- Residuals show good distribution
- No obvious non-linear patterns

#### 7. **Computational Efficiency**
- Training time: < 1 second
- Prediction time: < 1 millisecond
- Memory efficient
- Suitable for real-time applications

#### 8. **Educational Value**
- Fundamental algorithm for learning ML
- Clear mathematical foundation
- Easy to understand and implement
- Good baseline for comparison

---

## 📊 Advertising Dataset Analysis

### Feature-Target Relationships

```
TV Advertising:
- Range: 0.7 to 296.4 (thousands)
- Mean: 147.0
- Correlation with Sales: 0.78 (Strong)
- Interpretation: Strong positive linear relationship

Radio Advertising:
- Range: 0.0 to 49.6 (thousands)
- Mean: 23.3
- Correlation with Sales: 0.58 (Moderate)
- Interpretation: Moderate positive linear relationship

Newspaper Advertising:
- Range: 0.3 to 114.0 (thousands)
- Mean: 30.6
- Correlation with Sales: 0.23 (Weak)
- Interpretation: Weak positive linear relationship

Sales (Target):
- Range: 1.6 to 27.0 (thousands)
- Mean: 14.0
- Standard Deviation: 5.2
```

### Why Linear Model Fits Well

1. **Scatter Plot Analysis**: Points roughly follow a line
2. **Correlation Strength**: Strong correlations indicate linearity
3. **Residual Distribution**: Residuals appear randomly distributed
4. **No Obvious Patterns**: No curved or non-linear patterns visible
5. **Homoscedasticity**: Variance appears constant across range

---

## 🔬 Mathematical Derivation for Advertising

### Setting Up the Problem

**Model Equation**:
```
Sales = β₀ + β₁(TV) + β₂(Radio) + β₃(Newspaper) + ε
```

**Matrix Form**:
```
y = Xβ + ε

Where:
y = [Sales₁]      X = [1  TV₁  Radio₁  Newspaper₁]
    [Sales₂]          [1  TV₂  Radio₂  Newspaper₂]
    [⋮      ]         [⋮  ⋮    ⋮       ⋮          ]
    [Sales₂₀₀]        [1  TV₂₀₀ Radio₂₀₀ Newspaper₂₀₀]

β = [β₀]
    [β₁]
    [β₂]
    [β₃]
```

### Solving for Coefficients

**Objective**: Minimize SSE = Σ(yᵢ - ŷᵢ)²

**Solution**:
```
β = (XᵀX)⁻¹Xᵀy

Step 1: Calculate XᵀX (4×4 matrix)
Step 2: Calculate inverse (XᵀX)⁻¹
Step 3: Calculate Xᵀy (4×1 vector)
Step 4: Multiply (XᵀX)⁻¹Xᵀy to get β
```

### Example Coefficients

```
Typical Results:
β₀ = 6.97  (Intercept: baseline sales)
β₁ = 0.046 (TV coefficient: each $1000 TV spend → $46 sales increase)
β₂ = 0.189 (Radio coefficient: each $1000 Radio spend → $189 sales increase)
β₃ = -0.001 (Newspaper coefficient: minimal negative impact)
```

### Making Predictions

**For a New Campaign**:
```
Input: TV = $100k, Radio = $30k, Newspaper = $20k

Prediction:
Sales = 6.97 + 0.046(100) + 0.189(30) + (-0.001)(20)
      = 6.97 + 4.6 + 5.67 - 0.02
      = 17.21 (thousands = $17,210)
```

---

## 🔄 Training Process

### Step 1: Data Preparation
```
1. Load 200 samples from GitHub
2. Extract features: X = [TV, Radio, Newspaper]
3. Extract target: y = [Sales]
4. Add intercept column to X
```

### Step 2: Train-Test Split
```
Total: 200 samples
Training: 134 samples (67%)
Testing: 66 samples (33%)
Random state: 3 (reproducibility)
```

### Step 3: Feature Scaling
```
StandardScaler:
- Calculate mean and std for each feature
- Transform: x_scaled = (x - mean) / std
- Apply to both training and testing data
```

### Step 4: Model Training
```
Using Normal Equation:
β = (XᵀX)⁻¹Xᵀy

Computational complexity: O(n³) where n = number of features
For 3 features: Very fast computation
```

### Step 5: Evaluation
```
On Training Set:
- Calculate predictions: ŷ = Xβ
- Calculate metrics: MSE, RMSE, MAE, R²

On Testing Set:
- Calculate predictions: ŷ = Xβ
- Calculate metrics: MSE, RMSE, MAE, R²
- Compare with training metrics
```

---

## 📈 Performance Analysis

### Expected Results

```
Training Metrics:
- R² Score: 0.906 (90.6% variance explained)
- RMSE: 1.51 (average error: $1,510)
- MAE: 1.18 (mean absolute error: $1,180)

Testing Metrics:
- R² Score: 0.872 (87.2% variance explained)
- RMSE: 2.04 (average error: $2,040)
- MAE: 1.40 (mean absolute error: $1,400)

Interpretation:
- Model explains 87% of variance in test data
- Average prediction error: ±$2,040
- Good generalization (small gap between train/test)
```

### Residual Analysis

```
Residuals = Actual - Predicted

Properties of Good Residuals:
✓ Mean ≈ 0 (no systematic bias)
✓ Normally distributed
✓ Constant variance (homoscedasticity)
✓ No patterns or trends
✓ Independent observations
```

---

## 🎯 Practical Implications

### Business Insights

1. **TV Advertising**: Strongest impact on sales
   - Coefficient: 0.046
   - Interpretation: $1000 TV spend → $46 sales increase

2. **Radio Advertising**: Moderate impact
   - Coefficient: 0.189
   - Interpretation: $1000 Radio spend → $189 sales increase

3. **Newspaper Advertising**: Minimal impact
   - Coefficient: -0.001
   - Interpretation: Negligible or negative impact

### Decision Making

```
Budget Allocation Strategy:
- Prioritize TV advertising (highest ROI)
- Use Radio as secondary channel
- Minimize Newspaper spending
- Test different allocations using model
```

---

## 🔗 Comparison with Alternatives

### Why Not Other Algorithms?

| Algorithm | Reason Not Chosen |
|-----------|------------------|
| Polynomial Regression | No evidence of non-linear relationships |
| Ridge/Lasso Regression | No multicollinearity issues detected |
| Decision Trees | Overkill for linear data, less interpretable |
| Neural Networks | Too complex for simple linear relationship |
| SVM | Unnecessary for linear problem |

---

## 💡 Key Takeaways

1. **Linear Regression** is ideal for predicting continuous values with linear relationships
2. **Advertising Dataset** shows clear linear relationships between features and sales
3. **Mathematical Foundation** based on minimizing sum of squared errors
4. **Interpretability** allows business stakeholders to understand feature impact
5. **Performance** is strong (R² ≈ 0.87) with good generalization
6. **Efficiency** makes it suitable for real-time predictions
7. **Educational Value** as fundamental ML algorithm

---

## 📚 Further Reading

### Concepts to Explore
- Regularization (Ridge, Lasso)
- Polynomial Regression
- Gradient Descent vs Normal Equation
- Feature Engineering
- Cross-validation
- Residual Analysis

### Resources
- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Linear Regression Mathematics](https://en.wikipedia.org/wiki/Linear_regression)
- [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)

---

**Last Updated**: January 2026
**Version**: 1.0
**Status**: Complete
