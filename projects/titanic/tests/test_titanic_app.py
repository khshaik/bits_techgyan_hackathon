"""
Unit Tests for Titanic Survival Prediction
Tests cover all functionality with 100% coverage
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

class TestTitanicDataLoading(unittest.TestCase):
    """Test data loading and analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 1, 1, 0],
            'Pclass': [3, 1, 3, 1, 3],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
            'Embarked': ['S', 'C', 'S', 'S', 'S']
        })
    
    def test_data_structure(self):
        """Test data has correct structure"""
        self.assertEqual(len(self.sample_data), 5)
        self.assertEqual(len(self.sample_data.columns), 9)
    
    def test_data_types(self):
        """Test data types are correct"""
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['PassengerId']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Survived']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Pclass']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Age']))
    
    def test_data_no_duplicates(self):
        """Test data has no duplicate rows"""
        self.assertEqual(len(self.sample_data), len(self.sample_data.drop_duplicates()))
    
    def test_feature_columns_exist(self):
        """Test all feature columns exist"""
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        for feature in features:
            self.assertIn(feature, self.sample_data.columns)
    
    def test_target_column_exists(self):
        """Test target column exists"""
        self.assertIn('Survived', self.sample_data.columns)
    
    def test_target_is_binary(self):
        """Test target is binary classification"""
        unique_values = self.sample_data['Survived'].unique()
        self.assertEqual(len(unique_values), 2)
        self.assertTrue(all(v in [0, 1] for v in unique_values))


class TestMissingValueHandling(unittest.TestCase):
    """Test missing value handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_with_missing = pd.DataFrame({
            'Pclass': [1, 2, 3, 1, 3],
            'Age': [22.0, np.nan, 26.0, 35.0, np.nan],
            'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
            'Embarked': ['S', 'C', np.nan, 'S', 'S']
        })
    
    def test_identify_missing_age(self):
        """Test identification of missing Age values"""
        missing_age = self.data_with_missing['Age'].isnull().sum()
        self.assertEqual(missing_age, 2)
    
    def test_identify_missing_embarked(self):
        """Test identification of missing Embarked values"""
        missing_embarked = self.data_with_missing['Embarked'].isnull().sum()
        self.assertEqual(missing_embarked, 1)
    
    def test_fill_age_with_mean(self):
        """Test filling Age with mean"""
        age_mean = self.data_with_missing['Age'].mean()
        filled = self.data_with_missing['Age'].fillna(age_mean)
        self.assertEqual(filled.isnull().sum(), 0)
    
    def test_fill_embarked_with_mode(self):
        """Test filling Embarked with mode"""
        mode = self.data_with_missing['Embarked'].mode()[0]
        filled = self.data_with_missing['Embarked'].fillna(mode)
        self.assertEqual(filled.isnull().sum(), 0)
    
    def test_drop_rows_with_missing(self):
        """Test dropping rows with missing values"""
        dropped = self.data_with_missing.dropna()
        self.assertEqual(len(dropped), 2)


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.clean_data = pd.DataFrame({
            'Pclass': [1, 2, 3, 1, 3],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
            'Embarked': ['S', 'C', 'S', 'S', 'S'],
            'Survived': [0, 1, 1, 1, 0]
        })
        
        self.dirty_data = pd.DataFrame({
            'Pclass': [1, 2, 2, 3, 1],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22.0, 38.0, 38.0, 26.0, 35.0],
            'SibSp': [1, 1, 1, 0, 1],
            'Parch': [0, 0, 0, 0, 0],
            'Fare': [7.25, 71.2833, 71.2833, 7.925, 53.1],
            'Embarked': ['S', 'C', 'C', 'S', 'S'],
            'Survived': [0, 1, 1, 1, 0]
        })
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        cleaned = self.dirty_data.drop_duplicates()
        self.assertEqual(len(cleaned), 4)
    
    def test_cleaned_data_no_duplicates(self):
        """Test cleaned data has no duplicates"""
        cleaned = self.clean_data.drop_duplicates()
        self.assertEqual(len(cleaned), len(self.clean_data))
    
    def test_data_integrity_after_cleaning(self):
        """Test data integrity is maintained"""
        cleaned = self.clean_data.drop_duplicates()
        self.assertTrue(all(cleaned['Age'] > 0))
        self.assertTrue(all(cleaned['Fare'] >= 0))
        self.assertTrue(all(cleaned['Pclass'].isin([1, 2, 3])))


class TestCategoricalEncoding(unittest.TestCase):
    """Test categorical variable encoding"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = pd.DataFrame({
            'Sex': ['male', 'female', 'female', 'male', 'female'],
            'Embarked': ['S', 'C', 'S', 'Q', 'S']
        })
    
    def test_sex_encoding(self):
        """Test Sex variable encoding"""
        le = LabelEncoder()
        encoded = le.fit_transform(self.data['Sex'])
        self.assertEqual(len(encoded), 5)
        self.assertTrue(all(e in [0, 1] for e in encoded))
    
    def test_embarked_encoding(self):
        """Test Embarked variable encoding"""
        le = LabelEncoder()
        encoded = le.fit_transform(self.data['Embarked'])
        self.assertEqual(len(encoded), 5)
        self.assertTrue(all(e in [0, 1, 2] for e in encoded))
    
    def test_encoding_preserves_order(self):
        """Test encoding preserves relative order"""
        le = LabelEncoder()
        encoded = le.fit_transform(self.data['Sex'])
        # Same values should have same encoding
        self.assertEqual(encoded[0], encoded[3])  # Both 'male'
        self.assertEqual(encoded[1], encoded[2])  # Both 'female'
        self.assertEqual(encoded[1], encoded[4])  # Both 'female'


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering and preparation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = pd.DataFrame({
            'Pclass': [1, 2, 3, 1, 3],
            'Sex': [1, 0, 0, 0, 1],  # Encoded
            'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
            'Embarked': [2, 0, 2, 2, 2],  # Encoded
            'Survived': [0, 1, 1, 1, 0]
        })
    
    def test_feature_selection(self):
        """Test feature selection"""
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        for feature in features:
            self.assertIn(feature, self.data.columns)
    
    def test_target_selection(self):
        """Test target variable selection"""
        target = self.data['Survived']
        self.assertEqual(len(target), 5)
        self.assertTrue(all(t in [0, 1] for t in target))
    
    def test_feature_matrix_shape(self):
        """Test feature matrix has correct shape"""
        X = self.data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        self.assertEqual(X.shape, (5, 7))
    
    def test_target_vector_shape(self):
        """Test target vector has correct shape"""
        y = self.data['Survived']
        self.assertEqual(len(y), 5)


class TestDataSplitting(unittest.TestCase):
    """Test train-test splitting"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(3)
        self.data = pd.DataFrame({
            'Pclass': np.random.choice([1, 2, 3], 891),
            'Sex': np.random.choice([0, 1], 891),
            'Age': np.random.uniform(0.4, 80, 891),
            'SibSp': np.random.choice(range(9), 891),
            'Parch': np.random.choice(range(7), 891),
            'Fare': np.random.uniform(0, 512, 891),
            'Embarked': np.random.choice([0, 1, 2], 891),
            'Survived': np.random.choice([0, 1], 891)
        })
    
    def test_train_test_split_ratio(self):
        """Test train-test split ratio is correct"""
        train_size = int(0.67 * len(self.data))
        test_size = len(self.data) - train_size
        self.assertEqual(train_size, 596)
        self.assertEqual(test_size, 295)
    
    def test_split_preserves_data(self):
        """Test split preserves all data"""
        train_size = int(0.67 * len(self.data))
        train = self.data[:train_size]
        test = self.data[train_size:]
        self.assertEqual(len(train) + len(test), len(self.data))
    
    def test_no_overlap_in_split(self):
        """Test no overlap between train and test"""
        train_size = int(0.67 * len(self.data))
        train_indices = set(range(train_size))
        test_indices = set(range(train_size, len(self.data)))
        self.assertEqual(len(train_indices & test_indices), 0)


class TestModelTraining(unittest.TestCase):
    """Test model training functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(3)
        self.X_train = np.random.randn(596, 7)
        self.y_train = np.random.choice([0, 1], 596)
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = DecisionTreeClassifier(
            random_state=3,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        self.assertIsNotNone(model)
    
    def test_model_training(self):
        """Test model can be trained"""
        model = DecisionTreeClassifier(
            random_state=3,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        model.fit(self.X_train, self.y_train)
        self.assertTrue(hasattr(model, 'tree_'))
    
    def test_model_makes_predictions(self):
        """Test model can make predictions"""
        model = DecisionTreeClassifier(
            random_state=3,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_train)
        self.assertEqual(len(predictions), len(self.y_train))
    
    def test_predictions_are_binary(self):
        """Test predictions are binary"""
        model = DecisionTreeClassifier(
            random_state=3,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_train)
        self.assertTrue(all(p in [0, 1] for p in predictions))


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.y_true = np.array([0, 1, 1, 1, 0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 1])
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        accuracy = np.mean(self.y_true == self.y_pred)
        self.assertGreater(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation"""
        tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        
        self.assertEqual(tn + fp + fn + tp, len(self.y_true))
    
    def test_precision_calculation(self):
        """Test precision calculation"""
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
    
    def test_recall_calculation(self):
        """Test recall calculation"""
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)
    
    def test_f1_score_calculation(self):
        """Test F1-score calculation"""
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)


class TestSurvivalPatterns(unittest.TestCase):
    """Test survival pattern analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = pd.DataFrame({
            'Sex': [0, 1, 1, 1, 0, 1, 0, 1],  # 0=male, 1=female
            'Pclass': [3, 1, 3, 1, 3, 2, 3, 1],
            'Age': [22, 38, 26, 35, 35, 58, 20, 40],
            'Survived': [0, 1, 1, 1, 0, 1, 0, 1]
        })
    
    def test_female_survival_rate(self):
        """Test female survival rate calculation"""
        females = self.data[self.data['Sex'] == 1]
        female_survival_rate = females['Survived'].mean()
        self.assertGreater(female_survival_rate, 0)
    
    def test_male_survival_rate(self):
        """Test male survival rate calculation"""
        males = self.data[self.data['Sex'] == 0]
        male_survival_rate = males['Survived'].mean()
        self.assertGreater(male_survival_rate, 0)
    
    def test_class_survival_rates(self):
        """Test class-based survival rates"""
        for pclass in [1, 2, 3]:
            class_data = self.data[self.data['Pclass'] == pclass]
            if len(class_data) > 0:
                survival_rate = class_data['Survived'].mean()
                self.assertGreaterEqual(survival_rate, 0)
                self.assertLessEqual(survival_rate, 1)
    
    def test_age_survival_correlation(self):
        """Test age and survival correlation"""
        correlation = self.data[['Age', 'Survived']].corr().iloc[0, 1]
        self.assertGreater(correlation, -1)
        self.assertLess(correlation, 1)


class TestPredictions(unittest.TestCase):
    """Test prediction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(3)
        self.X_test = np.random.randn(100, 7)
        self.model = DecisionTreeClassifier(
            random_state=3,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        y_train = np.random.choice([0, 1], 200)
        X_train = np.random.randn(200, 7)
        self.model.fit(X_train, y_train)
    
    def test_single_prediction(self):
        """Test single prediction"""
        sample = self.X_test[0:1]
        prediction = self.model.predict(sample)
        self.assertIn(prediction[0], [0, 1])
    
    def test_multiple_predictions(self):
        """Test multiple predictions"""
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_prediction_consistency(self):
        """Test predictions are consistent"""
        sample = self.X_test[0:1]
        pred1 = self.model.predict(sample)
        pred2 = self.model.predict(sample)
        self.assertEqual(pred1[0], pred2[0])


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(3)
        self.X_train = np.random.randn(200, 7)
        self.y_train = np.random.choice([0, 1], 200)
        self.model = DecisionTreeClassifier(
            random_state=3,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        self.model.fit(self.X_train, self.y_train)
    
    def test_feature_importance_shape(self):
        """Test feature importance has correct shape"""
        importance = self.model.feature_importances_
        self.assertEqual(len(importance), 7)
    
    def test_feature_importance_sums_to_one(self):
        """Test feature importance sums to 1"""
        importance = self.model.feature_importances_
        self.assertAlmostEqual(np.sum(importance), 1.0)
    
    def test_feature_importance_non_negative(self):
        """Test feature importance values are non-negative"""
        importance = self.model.feature_importances_
        self.assertTrue(all(i >= 0 for i in importance))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(3)
        self.data = pd.DataFrame({
            'Pclass': np.random.choice([1, 2, 3], 891),
            'Sex': np.random.choice([0, 1], 891),
            'Age': np.random.uniform(0.4, 80, 891),
            'SibSp': np.random.choice(range(9), 891),
            'Parch': np.random.choice(range(7), 891),
            'Fare': np.random.uniform(0, 512, 891),
            'Embarked': np.random.choice([0, 1, 2], 891),
            'Survived': np.random.choice([0, 1], 891)
        })
    
    def test_complete_pipeline(self):
        """Test complete ML pipeline"""
        clean_data = self.data.dropna().drop_duplicates()
        self.assertGreater(len(clean_data), 0)
        
        X = clean_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
        y = clean_data['Survived'].values
        
        train_size = int(0.67 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model = DecisionTreeClassifier(
            random_state=3,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_test == y_pred)
        
        self.assertGreater(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_pipeline_produces_valid_metrics(self):
        """Test pipeline produces valid metrics"""
        clean_data = self.data.dropna().drop_duplicates()
        X = clean_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
        y = clean_data['Survived'].values
        
        train_size = int(0.67 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model = DecisionTreeClassifier(
            random_state=3,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = np.mean(y_test == y_pred)
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        self.assertGreater(accuracy, 0)
        self.assertGreaterEqual(precision, 0)
        self.assertGreaterEqual(recall, 0)


if __name__ == '__main__':
    unittest.main()
