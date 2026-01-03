# Decision Tree Classifier Algorithm - Titanic Survival Prediction

## 📚 Overview

Decision Tree Classifier is a supervised learning algorithm that makes predictions by learning simple decision rules from data. In the Titanic project, it predicts passenger survival by creating a tree of yes/no questions about passenger characteristics (class, gender, age, etc.).

---

## 🎯 Algorithm Significance

### Why Decision Trees Matter

1. **Interpretability**: Decision rules are easy to understand and visualize
2. **Non-linear Relationships**: Captures complex patterns without explicit formulas
3. **Feature Importance**: Automatically ranks features by importance
4. **No Scaling Required**: Works with raw feature values
5. **Mixed Data Types**: Handles both numerical and categorical features
6. **Explainability**: Can explain individual predictions
7. **Business-Friendly**: Non-technical stakeholders can understand decisions

### Key Characteristics

- **Supervised Learning**: Requires labeled training data
- **Classification Task**: Predicts discrete categories (survived/not survived)
- **Non-parametric**: No fixed number of parameters
- **Tree-based**: Hierarchical structure of decision nodes
- **Greedy Algorithm**: Makes locally optimal decisions at each split
- **Deterministic**: Same input always produces same output

---

## 🔧 How Decision Trees Work

### Conceptual Understanding

A Decision Tree asks a series of yes/no questions about features, creating a tree structure that partitions the data into increasingly pure groups. Each path from root to leaf represents a decision rule.

```
Example Decision Path:
                    Is Passenger Female?
                    /                  \
                  YES                   NO
                   |                     |
            Survived (High %)      Is Passenger Class 1?
                                   /              \
                                 YES              NO
                                  |                |
                          Survived (High %)   Did Not Survive (High %)
```

### Step-by-Step Process

#### Step 1: Start with All Data
```
Root Node: All 891 passengers
- Survived: 342 (38.4%)
- Did Not Survive: 549 (61.6%)
```

#### Step 2: Find Best Split
```
For each feature:
  For each possible split value:
    Calculate information gain
    
Select feature and value with highest information gain
```

#### Step 3: Create Child Nodes
```
Split data based on best feature
Create left child (feature ≤ threshold)
Create right child (feature > threshold)
```

#### Step 4: Recursively Split
```
Repeat Steps 2-3 for each child node
Until stopping criteria met:
  - Max depth reached
  - Min samples in node
  - Pure node (all same class)
  - No information gain
```

#### Step 5: Make Predictions
```
For new passenger:
  Start at root
  Follow decision rules down tree
  Reach leaf node
  Predict majority class in leaf
```

---

## 📐 Mathematical Formulas

### 1. Entropy (Information Content)

**Entropy Measures Impurity**:
```
H(S) = -Σ pᵢ × log₂(pᵢ)

Where:
- S: Dataset
- pᵢ: Proportion of class i in S
- log₂: Binary logarithm

Example (Titanic):
H(S) = -[0.384 × log₂(0.384) + 0.616 × log₂(0.616)]
     = -[0.384 × (-1.38) + 0.616 × (-0.70)]
     = 0.531 + 0.431
     = 0.962 bits

Interpretation:
- H = 0: Pure node (all same class)
- H = 1: Maximum impurity (equal split)
```

### 2. Information Gain

**Information Gain Measures Split Quality**:
```
IG(S, A) = H(S) - Σ |Sᵥ|/|S| × H(Sᵥ)

Where:
- S: Parent dataset
- A: Attribute (feature) to split on
- Sᵥ: Subset where attribute = v
- H(S): Entropy of parent
- H(Sᵥ): Entropy of child v

Higher IG = Better split
```

### 3. Gini Impurity

**Alternative to Entropy** (used in scikit-learn):
```
Gini(S) = 1 - Σ pᵢ²

Where:
- pᵢ: Proportion of class i

Example (Titanic):
Gini(S) = 1 - [0.384² + 0.616²]
        = 1 - [0.147 + 0.379]
        = 1 - 0.526
        = 0.474

Interpretation:
- Gini = 0: Pure node
- Gini = 0.5: Maximum impurity (binary classification)
```

### 4. Weighted Gini After Split

**Gini for Split Quality**:
```
Gini_split = (n_left/n_total) × Gini_left + (n_right/n_total) × Gini_right

Where:
- n_left: Samples in left child
- n_right: Samples in right child
- n_total: Total samples

Lower Gini_split = Better split
```

### 5. Information Gain (Alternative Formula)

**Using Gini**:
```
IG = Gini_parent - Gini_split
```

### 6. Classification Metrics

**Accuracy** (Overall correctness):
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Where:
- TP: True Positives (correctly predicted survivors)
- TN: True Negatives (correctly predicted non-survivors)
- FP: False Positives (incorrectly predicted survivors)
- FN: False Negatives (incorrectly predicted non-survivors)
```

**Precision** (Of predicted survivors, how many actually survived):
```
Precision = TP / (TP + FP)

Interpretation:
- High precision: Few false alarms
- Low precision: Many false positives
```

**Recall** (Of actual survivors, how many were correctly predicted):
```
Recall = TP / (TP + FN)

Interpretation:
- High recall: Catches most survivors
- Low recall: Misses many survivors
```

**F1-Score** (Harmonic mean of precision and recall):
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Interpretation:
- Balances precision and recall
- 0 to 1 scale (1 is perfect)
```

### 7. Confusion Matrix

**Binary Classification Matrix**:
```
                Predicted Survived    Predicted Did Not Survive
Actually Survived:      TP                      FN
Actually Did Not Survive: FP                    TN
```

### 8. Feature Importance

**Importance Based on Splits**:
```
Importance(feature) = (Σ gain from splits using feature) / (Total gain)

Normalized to sum to 1.0

Example:
Sex: 0.45 (45% of total importance)
Pclass: 0.35 (35% of total importance)
Age: 0.15 (15% of total importance)
Fare: 0.05 (5% of total importance)
```

---

## 🎓 Learning Perspective

### Advantages of Decision Trees

✓ Highly interpretable and explainable
✓ Requires no feature scaling
✓ Handles both numerical and categorical data
✓ Captures non-linear relationships
✓ Provides feature importance rankings
✓ Fast predictions (logarithmic time)
✓ Can handle missing values
✓ Works with imbalanced datasets
✓ No assumptions about data distribution

### Disadvantages of Decision Trees

✗ Prone to overfitting (especially deep trees)
✗ Unstable (small data changes → big tree changes)
✗ Biased toward high-cardinality features
✗ Can create biased trees with imbalanced classes
✗ Greedy algorithm (not globally optimal)
✗ May create unnecessarily complex trees
✗ Requires pruning to generalize well

### Hyperparameters to Control

```
max_depth: Maximum tree depth
  - Smaller: Simpler model, less overfitting
  - Larger: More complex model, better training fit

min_samples_split: Minimum samples to split node
  - Larger: Simpler model, less overfitting
  - Smaller: More complex model

min_samples_leaf: Minimum samples in leaf node
  - Larger: Simpler model, smoother predictions
  - Smaller: More complex model

criterion: Split criterion (gini or entropy)
  - Usually similar results
```

---

## 🎯 Why Decision Tree for Titanic Dataset

### Dataset Characteristics

```
Dataset: Titanic Survival
- Samples: 891 passengers
- Features: 11 (mix of numerical and categorical)
- Target: Survived (binary: 0 or 1)
- Class Distribution: 38.4% survived, 61.6% did not
- Feature Types: Age (numerical), Sex (categorical), Pclass (categorical), etc.
```

### Reasons for Selection

#### 1. **Mixed Feature Types**
```
Categorical Features:
- Sex: Male/Female (binary)
- Embarked: C/Q/S (3 categories)
- Pclass: 1/2/3 (3 categories)

Numerical Features:
- Age: 0-80 years
- Fare: 0-512 dollars
- SibSp: 0-8 relatives
- Parch: 0-6 relatives

Decision Trees handle both naturally without encoding
```

#### 2. **Non-linear Relationships**
```
Survival Patterns:
- Gender: "Women and children first" policy
- Class: First-class had much higher survival
- Age: Children had higher survival
- Fare: Higher fare → Higher survival (proxy for class)

These are non-linear patterns that Decision Trees capture well
```

#### 3. **Interpretability Requirement**
```
Historical Context:
- Want to understand survival factors
- Need to explain decisions to historians/researchers
- Decision Trees provide clear decision rules

Example Rule:
IF Female AND (Class = 1 OR Class = 2) THEN Survived
IF Male AND Fare > 50 THEN Survived
```

#### 4. **Feature Importance Analysis**
```
Business Question: Which factors most influenced survival?

Decision Tree Answer:
- Sex: 45% importance (strongest factor)
- Pclass: 35% importance
- Age: 15% importance
- Fare: 5% importance

Clear ranking of survival factors
```

#### 5. **No Scaling Required**
```
Advantages:
- Age (0-80) and Fare (0-512) on different scales
- Decision Trees don't need scaling
- Reduces preprocessing steps
- Faster implementation
```

#### 6. **Handling Missing Values**
```
Dataset has missing values:
- Age: 177 missing (19.9%)
- Cabin: 687 missing (77.1%)
- Embarked: 2 missing (0.2%)

Decision Trees can:
- Handle missing values naturally
- Use surrogate splits
- Not require imputation
```

#### 7. **Class Imbalance Handling**
```
Survival Distribution:
- Survived: 342 (38.4%)
- Did Not Survive: 549 (61.6%)

Decision Trees:
- Can use class weights
- Can adjust min_samples_leaf
- Naturally handle imbalance
```

#### 8. **Prediction Speed**
```
Inference Time: O(log n) where n = tree depth
- Very fast predictions
- Suitable for real-time applications
- Efficient for batch processing
```

---

## 📊 Titanic Dataset Analysis

### Feature Analysis

```
Sex (Gender):
- Female: 314 passengers, 74.2% survived
- Male: 577 passengers, 18.9% survived
→ Strongest survival predictor

Pclass (Passenger Class):
- Class 1: 216 passengers, 63.0% survived
- Class 2: 184 passengers, 47.3% survived
- Class 3: 491 passengers, 24.2% survived
→ Clear class-based survival disparity

Age:
- Children (< 10): Higher survival rate
- Adults (20-50): Lower survival rate
- Elderly (> 60): Variable survival
→ Age affects survival chances

Fare (Ticket Price):
- High fare: Higher survival (proxy for class)
- Low fare: Lower survival
→ Economic status indicator

SibSp (Siblings/Spouses):
- Traveling alone: Lower survival
- Traveling with 1-2 relatives: Higher survival
- Traveling with 3+ relatives: Lower survival
→ Family size affects survival
```

### Why Decision Tree Fits Well

1. **Clear Decision Boundaries**: Sex and class create clear survival groups
2. **Non-linear Patterns**: "Women and children first" is non-linear
3. **Feature Interactions**: Class affects survival differently for men vs women
4. **Categorical Features**: Sex and Embarked are naturally categorical
5. **Interpretable Results**: Easy to explain historical patterns

---

## 🔬 Mathematical Derivation for Titanic

### Building the Tree

#### Step 1: Calculate Root Node Entropy
```
Root Node: 891 passengers
- Survived: 342 (38.4%)
- Did Not Survive: 549 (61.6%)

H(Root) = -[0.384 × log₂(0.384) + 0.616 × log₂(0.616)]
        = 0.962 bits
```

#### Step 2: Evaluate Splits

**Split on Sex**:
```
Female (314 passengers):
- Survived: 233 (74.2%)
- Did Not Survive: 81 (25.8%)
- H(Female) = 0.806 bits

Male (577 passengers):
- Survived: 109 (18.9%)
- Did Not Survive: 468 (81.1%)
- H(Male) = 0.668 bits

Weighted Entropy:
H(Sex) = (314/891) × 0.806 + (577/891) × 0.668
       = 0.283 + 0.432
       = 0.715 bits

Information Gain:
IG(Sex) = 0.962 - 0.715 = 0.247 bits (High!)
```

**Split on Pclass**:
```
Class 1 (216 passengers):
- Survived: 136 (63.0%)
- Did Not Survive: 80 (37.0%)
- H(Class1) = 0.956 bits

Class 2 (184 passengers):
- Survived: 87 (47.3%)
- Did Not Survive: 97 (52.7%)
- H(Class2) = 0.998 bits

Class 3 (491 passengers):
- Survived: 119 (24.2%)
- Did Not Survive: 372 (75.8%)
- H(Class3) = 0.811 bits

Weighted Entropy:
H(Pclass) = (216/891) × 0.956 + (184/891) × 0.998 + (491/891) × 0.811
          = 0.233 + 0.206 + 0.447
          = 0.886 bits

Information Gain:
IG(Pclass) = 0.962 - 0.886 = 0.076 bits (Lower than Sex)
```

#### Step 3: Select Best Split
```
IG(Sex) = 0.247 bits ← HIGHEST
IG(Pclass) = 0.076 bits
IG(Age) = 0.065 bits
IG(Fare) = 0.051 bits

→ Split on Sex (Female/Male)
```

#### Step 4: Recursively Split Children

**Left Child (Female)**:
```
314 passengers, 74.2% survived
Next best split: Pclass
- Class 1: 91 passengers, 96.7% survived
- Class 2: 70 passengers, 69.0% survived
- Class 3: 153 passengers, 50.3% survived
```

**Right Child (Male)**:
```
577 passengers, 18.9% survived
Next best split: Pclass
- Class 1: 125 passengers, 36.8% survived
- Class 2: 114 passengers, 16.7% survived
- Class 3: 338 passengers, 13.5% survived
```

#### Step 5: Continue Until Stopping Criteria
```
Stopping Criteria:
- Max depth: 5 (prevent overfitting)
- Min samples split: 10 (minimum to split)
- Min samples leaf: 5 (minimum in leaf)
```

---

## 🔄 Training Process

### Step 1: Data Preparation
```
1. Load 891 passenger records
2. Handle missing values:
   - Age: Fill with class-specific mean
   - Embarked: Fill with mode 'S'
   - Cabin: Drop (too many missing)
3. Encode categorical variables:
   - Sex: Male=0, Female=1
   - Embarked: C=0, Q=1, S=2
4. Select features: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
```

### Step 2: Train-Test Split
```
Total: 891 passengers
Training: 596 passengers (67%)
Testing: 295 passengers (33%)
Random state: 3 (reproducibility)
```

### Step 3: Model Training
```
DecisionTreeClassifier(
    criterion='gini',           # Split criterion
    max_depth=5,                # Prevent overfitting
    min_samples_split=10,       # Minimum to split
    min_samples_leaf=5,         # Minimum in leaf
    random_state=3              # Reproducibility
)

Training Algorithm:
1. Start with root node (all 891 passengers)
2. For each node:
   a. Calculate Gini for each feature
   b. Find feature with lowest Gini split
   c. Create left and right children
   d. Recursively apply to children
3. Stop when criteria met
```

### Step 4: Evaluation
```
On Training Set:
- Make predictions: ŷ = tree.predict(X_train)
- Calculate metrics: Accuracy, Precision, Recall, F1

On Testing Set:
- Make predictions: ŷ = tree.predict(X_test)
- Calculate metrics: Accuracy, Precision, Recall, F1
- Generate confusion matrix
```

---

## 📈 Performance Analysis

### Expected Results

```
Training Metrics:
- Accuracy: 78-82%
- Precision: 72-76%
- Recall: 73-77%
- F1-Score: 72-76%

Testing Metrics:
- Accuracy: 78-82%
- Precision: 72-76%
- Recall: 73-77%
- F1-Score: 72-76%

Confusion Matrix (Testing):
                Predicted Survived    Predicted Did Not Survive
Actually Survived:      ~85                      ~30
Actually Did Not Survive: ~33                    ~147

Interpretation:
- Catches ~74% of actual survivors (Recall)
- 72% of predicted survivors actually survived (Precision)
- Overall 79% accuracy
```

### Feature Importance

```
Feature Importance Ranking:
Sex:      45% (Strongest factor)
Pclass:   35% (Second strongest)
Age:      15% (Moderate factor)
Fare:      5% (Weak factor)

Interpretation:
- Gender was the dominant survival factor
- Passenger class strongly influenced survival
- Age had moderate influence
- Ticket fare had minimal direct impact
```

---

## 🎯 Practical Implications

### Historical Insights

1. **"Women and Children First" Policy**
   - Clearly visible in tree structure
   - Female passengers had 74.2% survival rate
   - Male passengers had 18.9% survival rate

2. **Class-Based Survival**
   - First-class: 63.0% survival
   - Second-class: 47.3% survival
   - Third-class: 24.2% survival
   - Clear class discrimination

3. **Age Factor**
   - Children prioritized in evacuation
   - Younger passengers had better chances
   - Elderly had lower survival rates

4. **Economic Status**
   - Higher fares → Higher survival
   - Proxy for class and cabin location
   - Better access to lifeboats

### Decision Rules

```
Rule 1: IF Female AND (Class = 1 OR Class = 2) THEN Survived (High probability)
Rule 2: IF Female AND Class = 3 AND Age < 10 THEN Survived (Moderate probability)
Rule 3: IF Male AND Class = 1 AND Fare > 50 THEN Survived (Moderate probability)
Rule 4: IF Male AND (Class = 2 OR Class = 3) THEN Did Not Survive (High probability)
```

---

## 🔗 Comparison with Alternatives

### Why Not Other Algorithms?

| Algorithm | Reason Not Chosen |
|-----------|------------------|
| Logistic Regression | Cannot capture non-linear patterns |
| SVM | Less interpretable, requires scaling |
| Neural Networks | Overkill, black-box nature |
| Random Forest | Good but less interpretable than single tree |
| Naive Bayes | Assumes feature independence |

---

## 💡 Key Takeaways

1. **Decision Trees** excel at capturing non-linear patterns and feature interactions
2. **Titanic Dataset** has clear non-linear survival patterns (gender, class effects)
3. **Mathematical Foundation** based on entropy and information gain
4. **Interpretability** allows understanding of historical survival factors
5. **Performance** is strong (78-82% accuracy) with good generalization
6. **Feature Importance** clearly ranks survival factors
7. **No Scaling Required** simplifies preprocessing

---

## 📚 Further Reading

### Concepts to Explore
- Tree Pruning (reduce overfitting)
- Ensemble Methods (Random Forest, Gradient Boosting)
- Cross-validation for hyperparameter tuning
- Handling Imbalanced Classes
- Feature Interactions
- Cost-sensitive Learning

### Resources
- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)
- [Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)

---

**Last Updated**: January 2026
**Version**: 1.0
**Status**: Complete
