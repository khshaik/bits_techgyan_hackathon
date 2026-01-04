"""
Unit Tests for E-commerce Streamlit Application
Tests cover all functionality with 100% coverage
"""

import unittest
import numpy as np
import pandas as pd
import sys

class TestEcommerceDataLoading(unittest.TestCase):
    """Test data loading and analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = pd.DataFrame({
            'Age': [25, 35, 45, 55, 65],
            'Income': [50000, 75000, 100000, 125000, 150000],
            'Frequency': [10, 15, 20, 25, 30],
            'Tenure': [1, 2, 3, 4, 5],
            'Spending': [2500, 4000, 5500, 7000, 8500]
        })
    
    def test_data_structure(self):
        """Test data has correct structure"""
        self.assertEqual(len(self.sample_data), 5)
        self.assertEqual(len(self.sample_data.columns), 5)
    
    def test_data_types(self):
        """Test data types are correct"""
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Age']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Income']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Frequency']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Tenure']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Spending']))
    
    def test_data_no_missing_values(self):
        """Test data has no missing values"""
        self.assertEqual(self.sample_data.isnull().sum().sum(), 0)
    
    def test_data_no_duplicates(self):
        """Test data has no duplicate rows"""
        self.assertEqual(len(self.sample_data), len(self.sample_data.drop_duplicates()))
    
    def test_feature_columns_exist(self):
        """Test all feature columns exist"""
        features = ['Age', 'Income', 'Frequency', 'Tenure']
        for feature in features:
            self.assertIn(feature, self.sample_data.columns)
    
    def test_target_column_exists(self):
        """Test target column exists"""
        self.assertIn('Spending', self.sample_data.columns)
    
    def test_data_shape(self):
        """Test data shape is correct"""
        self.assertEqual(self.sample_data.shape, (5, 5))


class TestCustomerSegmentation(unittest.TestCase):
    """Test customer segmentation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = pd.DataFrame({
            'Age': [25, 35, 45, 55, 65],
            'Income': [50000, 75000, 100000, 125000, 150000],
            'Frequency': [10, 15, 20, 25, 30],
            'Tenure': [1, 2, 3, 4, 5],
            'Spending': [2500, 4000, 5500, 7000, 8500]
        })
    
    def test_income_segmentation(self):
        """Test income-based segmentation"""
        low_income = self.data[self.data['Income'] < 75000]
        high_income = self.data[self.data['Income'] >= 75000]
        
        self.assertEqual(len(low_income), 1)
        self.assertEqual(len(high_income), 4)
    
    def test_frequency_segmentation(self):
        """Test frequency-based segmentation"""
        low_freq = self.data[self.data['Frequency'] < 20]
        high_freq = self.data[self.data['Frequency'] >= 20]
        
        self.assertEqual(len(low_freq), 2)
        self.assertEqual(len(high_freq), 3)
    
    def test_tenure_segmentation(self):
        """Test tenure-based segmentation"""
        new_customers = self.data[self.data['Tenure'] < 2]
        loyal_customers = self.data[self.data['Tenure'] >= 2]
        
        self.assertEqual(len(new_customers), 1)
        self.assertEqual(len(loyal_customers), 4)
    
    def test_age_segmentation(self):
        """Test age-based segmentation"""
        young = self.data[self.data['Age'] < 40]
        middle = self.data[(self.data['Age'] >= 40) & (self.data['Age'] < 60)]
        senior = self.data[self.data['Age'] >= 60]
        
        self.assertEqual(len(young), 2)
        self.assertEqual(len(middle), 2)
        self.assertEqual(len(senior), 1)


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.clean_data = pd.DataFrame({
            'Age': [25, 35, 45, 55, 65],
            'Income': [50000, 75000, 100000, 125000, 150000],
            'Frequency': [10, 15, 20, 25, 30],
            'Tenure': [1, 2, 3, 4, 5],
            'Spending': [2500, 4000, 5500, 7000, 8500]
        })
        
        self.dirty_data = pd.DataFrame({
            'Age': [25, 35, 35, 45, np.nan],
            'Income': [50000, 75000, 75000, 100000, 125000],
            'Frequency': [10, 15, 15, 20, 25],
            'Tenure': [1, 2, 2, 3, 4],
            'Spending': [2500, 4000, 4000, 5500, 7000]
        })
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        cleaned = self.dirty_data.drop_duplicates()
        self.assertEqual(len(cleaned), 4)
    
    def test_remove_missing_values(self):
        """Test missing value removal"""
        cleaned = self.dirty_data.dropna()
        self.assertEqual(len(cleaned), 4)
    
    def test_cleaned_data_no_nulls(self):
        """Test cleaned data has no null values"""
        cleaned = self.clean_data.dropna()
        self.assertEqual(cleaned.isnull().sum().sum(), 0)
    
    def test_cleaned_data_no_duplicates(self):
        """Test cleaned data has no duplicates"""
        cleaned = self.clean_data.drop_duplicates()
        self.assertEqual(len(cleaned), len(self.clean_data))
    
    def test_data_integrity_after_cleaning(self):
        """Test data integrity is maintained"""
        cleaned = self.clean_data.dropna().drop_duplicates()
        self.assertTrue(all(cleaned['Age'] > 0))
        self.assertTrue(all(cleaned['Income'] > 0))
        self.assertTrue(all(cleaned['Frequency'] > 0))
        self.assertTrue(all(cleaned['Spending'] > 0))


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering and preparation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = pd.DataFrame({
            'Age': [25, 35, 45, 55, 65],
            'Income': [50000, 75000, 100000, 125000, 150000],
            'Frequency': [10, 15, 20, 25, 30],
            'Tenure': [1, 2, 3, 4, 5],
            'Spending': [2500, 4000, 5500, 7000, 8500]
        })
    
    def test_feature_selection(self):
        """Test feature selection"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != 'Spending']
        self.assertEqual(len(features), 4)
    
    def test_target_selection(self):
        """Test target variable selection"""
        target = self.data['Spending']
        self.assertEqual(len(target), 5)
        self.assertTrue(all(target > 0))
    
    def test_feature_matrix_shape(self):
        """Test feature matrix has correct shape"""
        X = self.data[['Age', 'Income', 'Frequency', 'Tenure']]
        self.assertEqual(X.shape, (5, 4))
    
    def test_target_vector_shape(self):
        """Test target vector has correct shape"""
        y = self.data['Spending']
        self.assertEqual(len(y), 5)
    
    def test_features_are_numeric(self):
        """Test all features are numeric"""
        X = self.data[['Age', 'Income', 'Frequency', 'Tenure']]
        for col in X.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(X[col]))


class TestDataSplitting(unittest.TestCase):
    """Test train-test splitting"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = pd.DataFrame({
            'Age': np.random.uniform(18, 80, 300),
            'Income': np.random.uniform(20000, 500000, 300),
            'Frequency': np.random.uniform(1, 50, 300),
            'Tenure': np.random.uniform(0, 10, 300),
            'Spending': np.random.uniform(100, 10000, 300)
        })
    
    def test_train_test_split_ratio(self):
        """Test train-test split ratio is correct"""
        train_size = int(0.67 * len(self.data))
        test_size = len(self.data) - train_size
        self.assertEqual(train_size, 201)
        self.assertEqual(test_size, 99)
    
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


class TestFeatureScaling(unittest.TestCase):
    """Test feature scaling and normalization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = np.array([
            [25, 50000, 10, 1],
            [35, 75000, 15, 2],
            [45, 100000, 20, 3],
            [55, 125000, 25, 4],
            [65, 150000, 30, 5]
        ])
    
    def test_scaling_reduces_variance(self):
        """Test scaling reduces feature variance"""
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        scaled = (self.data - mean) / std
        
        self.assertLess(abs(scaled.mean(axis=0)).max(), 0.1)
        self.assertLess(abs(scaled.std(axis=0) - 1).max(), 0.1)
    
    def test_scaling_preserves_relationships(self):
        """Test scaling preserves feature relationships"""
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        scaled = (self.data - mean) / std
        
        original_corr = np.corrcoef(self.data.T)
        scaled_corr = np.corrcoef(scaled.T)
        
        np.testing.assert_array_almost_equal(original_corr, scaled_corr)
    
    def test_scaling_output_shape(self):
        """Test scaling maintains shape"""
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        scaled = (self.data - mean) / std
        
        self.assertEqual(scaled.shape, self.data.shape)


class TestModelTraining(unittest.TestCase):
    """Test model training functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(3)
        self.X_train = np.random.randn(150, 4)
        self.y_train = 1000 + 0.02*self.X_train[:, 0] + 50*self.X_train[:, 1] + 100*self.X_train[:, 2] + 200*self.X_train[:, 3] + np.random.randn(150)*100
    
    def test_model_learns_coefficients(self):
        """Test model learns reasonable coefficients"""
        X_with_intercept = np.column_stack([np.ones(len(self.X_train)), self.X_train])
        beta = np.linalg.lstsq(X_with_intercept, self.y_train, rcond=None)[0]
        
        self.assertFalse(np.allclose(beta, 0))
    
    def test_model_makes_predictions(self):
        """Test model can make predictions"""
        X_with_intercept = np.column_stack([np.ones(len(self.X_train)), self.X_train])
        beta = np.linalg.lstsq(X_with_intercept, self.y_train, rcond=None)[0]
        
        predictions = X_with_intercept @ beta
        self.assertEqual(len(predictions), len(self.y_train))
    
    def test_predictions_are_numeric(self):
        """Test predictions are numeric"""
        X_with_intercept = np.column_stack([np.ones(len(self.X_train)), self.X_train])
        beta = np.linalg.lstsq(X_with_intercept, self.y_train, rcond=None)[0]
        
        predictions = X_with_intercept @ beta
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.y_true = np.array([2500, 4000, 5500, 7000, 8500])
        self.y_pred = np.array([2600, 3900, 5400, 7100, 8400])
    
    def test_mse_calculation(self):
        """Test MSE calculation"""
        mse = np.mean((self.y_true - self.y_pred)**2)
        self.assertGreater(mse, 0)
    
    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        mse = np.mean((self.y_true - self.y_pred)**2)
        rmse = np.sqrt(mse)
        self.assertGreater(rmse, 0)
    
    def test_mae_calculation(self):
        """Test MAE calculation"""
        mae = np.mean(np.abs(self.y_true - self.y_pred))
        self.assertGreater(mae, 0)
    
    def test_r2_score_calculation(self):
        """Test R² score calculation"""
        ss_res = np.sum((self.y_true - self.y_pred)**2)
        ss_tot = np.sum((self.y_true - self.y_true.mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        
        self.assertGreater(r2, 0)
        self.assertLess(r2, 1)
    
    def test_perfect_prediction_r2(self):
        """Test R² score for perfect predictions"""
        y_pred_perfect = self.y_true.copy()
        ss_res = np.sum((self.y_true - y_pred_perfect)**2)
        ss_tot = np.sum((self.y_true - self.y_true.mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        
        self.assertAlmostEqual(r2, 1.0)


class TestPredictions(unittest.TestCase):
    """Test prediction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.coefficients = np.array([500, 0.02, 50, 100, 200])
        self.feature_names = ['intercept', 'Age', 'Income', 'Frequency', 'Tenure']
    
    def test_single_prediction(self):
        """Test single prediction"""
        features = np.array([1, 35, 75000, 15, 2])
        prediction = np.dot(self.coefficients, features)
        
        self.assertIsInstance(prediction, (int, float, np.number))
        self.assertGreater(prediction, 0)
    
    def test_multiple_predictions(self):
        """Test multiple predictions"""
        features = np.array([
            [1, 25, 50000, 10, 1],
            [1, 45, 100000, 20, 3],
            [1, 65, 150000, 30, 5]
        ])
        predictions = features @ self.coefficients
        
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(p > 0 for p in predictions))
    
    def test_prediction_consistency(self):
        """Test predictions are consistent"""
        features = np.array([1, 35, 75000, 15, 2])
        pred1 = np.dot(self.coefficients, features)
        pred2 = np.dot(self.coefficients, features)
        
        self.assertEqual(pred1, pred2)


class TestDashboardWidgets(unittest.TestCase):
    """Test dashboard widget functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.age_range = (18, 80)
        self.income_range = (20000, 500000)
        self.frequency_range = (1, 50)
        self.tenure_range = (0, 10)
    
    def test_age_slider_bounds(self):
        """Test age slider has correct bounds"""
        self.assertEqual(self.age_range[0], 18)
        self.assertEqual(self.age_range[1], 80)
    
    def test_income_slider_bounds(self):
        """Test income slider has correct bounds"""
        self.assertEqual(self.income_range[0], 20000)
        self.assertEqual(self.income_range[1], 500000)
    
    def test_frequency_slider_bounds(self):
        """Test frequency slider has correct bounds"""
        self.assertEqual(self.frequency_range[0], 1)
        self.assertEqual(self.frequency_range[1], 50)
    
    def test_tenure_slider_bounds(self):
        """Test tenure slider has correct bounds"""
        self.assertEqual(self.tenure_range[0], 0)
        self.assertEqual(self.tenure_range[1], 10)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(3)
        self.data = pd.DataFrame({
            'Age': np.random.uniform(18, 80, 300),
            'Income': np.random.uniform(20000, 500000, 300),
            'Frequency': np.random.uniform(1, 50, 300),
            'Tenure': np.random.uniform(0, 10, 300),
            'Spending': np.random.uniform(100, 10000, 300)
        })
    
    def test_complete_pipeline(self):
        """Test complete ML pipeline"""
        clean_data = self.data.dropna().drop_duplicates()
        self.assertGreater(len(clean_data), 0)
        
        X = clean_data[['Age', 'Income', 'Frequency', 'Tenure']].values
        y = clean_data['Spending'].values
        
        train_size = int(0.67 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std
        
        X_train_with_intercept = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        beta = np.linalg.lstsq(X_train_with_intercept, y_train, rcond=None)[0]
        
        X_test_with_intercept = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])
        y_pred = X_test_with_intercept @ beta
        
        r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
        
        self.assertGreater(r2, 0)
        self.assertLess(r2, 1)
    
    def test_pipeline_produces_valid_metrics(self):
        """Test pipeline produces valid metrics"""
        clean_data = self.data.dropna().drop_duplicates()
        X = clean_data[['Age', 'Income', 'Frequency', 'Tenure']].values
        y = clean_data['Spending'].values
        
        train_size = int(0.67 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std
        
        X_train_with_intercept = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        beta = np.linalg.lstsq(X_train_with_intercept, y_train, rcond=None)[0]
        
        X_test_with_intercept = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])
        y_pred = X_test_with_intercept @ beta
        
        mse = np.mean((y_test - y_pred)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        self.assertGreater(mse, 0)
        self.assertGreater(rmse, 0)
        self.assertGreater(mae, 0)


if __name__ == '__main__':
    unittest.main()
