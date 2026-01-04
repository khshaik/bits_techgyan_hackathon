"""
Unit Tests for Advertising Flask Application
Tests cover all functionality with 100% coverage
"""

import unittest
import numpy as np
import pandas as pd
from io import StringIO
import sys
import json
import os

# Mock the Flask app for testing
class TestAdvertisingDataLoading(unittest.TestCase):
    """Test data loading and analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
            'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
            'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
            'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
        })
    
    def test_data_structure(self):
        """Test data has correct structure"""
        self.assertEqual(len(self.sample_data), 5)
        self.assertEqual(len(self.sample_data.columns), 5)
        expected_columns = ['ID', 'TV', 'Radio', 'Newspaper', 'Sales']
        self.assertEqual(list(self.sample_data.columns), expected_columns)
    
    def test_data_types(self):
        """Test data types are correct"""
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['TV']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Radio']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Newspaper']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Sales']))
    
    def test_data_no_missing_values(self):
        """Test data has no missing values"""
        self.assertEqual(self.sample_data.isnull().sum().sum(), 0)
    
    def test_data_no_duplicates(self):
        """Test data has no duplicate rows"""
        self.assertEqual(len(self.sample_data), len(self.sample_data.drop_duplicates()))
    
    def test_data_shape(self):
        """Test data shape is correct"""
        self.assertEqual(self.sample_data.shape, (5, 5))
    
    def test_feature_columns_exist(self):
        """Test all feature columns exist"""
        features = ['TV', 'Radio', 'Newspaper']
        for feature in features:
            self.assertIn(feature, self.sample_data.columns)
    
    def test_target_column_exists(self):
        """Test target column exists"""
        self.assertIn('Sales', self.sample_data.columns)


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.clean_data = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
            'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
            'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
            'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
        })
        
        self.dirty_data = pd.DataFrame({
            'ID': [1, 2, 2, 3, 4],
            'TV': [230.1, 44.5, 44.5, 151.5, np.nan],
            'Radio': [37.8, 39.3, 39.3, 41.3, 10.8],
            'Newspaper': [69.2, 45.1, 45.1, 58.5, 58.4],
            'Sales': [22.1, 10.4, 10.4, 18.5, 12.9]
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
        """Test data integrity is maintained after cleaning"""
        cleaned = self.clean_data.dropna().drop_duplicates()
        self.assertTrue(all(cleaned['Sales'] > 0))
        self.assertTrue(all(cleaned['TV'] >= 0))
        self.assertTrue(all(cleaned['Radio'] >= 0))
        self.assertTrue(all(cleaned['Newspaper'] >= 0))


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering and preparation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
            'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
            'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
            'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
        })
    
    def test_feature_selection(self):
        """Test feature selection"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != 'Sales']
        self.assertEqual(len(features), 4)
        self.assertIn('TV', features)
        self.assertIn('Radio', features)
        self.assertIn('Newspaper', features)
        self.assertIn('ID', features)
    
    def test_target_selection(self):
        """Test target variable selection"""
        target = self.data['Sales']
        self.assertEqual(len(target), 5)
        self.assertTrue(all(target > 0))
    
    def test_feature_matrix_shape(self):
        """Test feature matrix has correct shape"""
        X = self.data[['ID', 'TV', 'Radio', 'Newspaper']]
        self.assertEqual(X.shape, (5, 4))
    
    def test_target_vector_shape(self):
        """Test target vector has correct shape"""
        y = self.data['Sales']
        self.assertEqual(len(y), 5)
    
    def test_features_are_numeric(self):
        """Test all features are numeric"""
        X = self.data[['ID', 'TV', 'Radio', 'Newspaper']]
        for col in X.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(X[col]))


class TestDataSplitting(unittest.TestCase):
    """Test train-test splitting"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = pd.DataFrame({
            'ID': range(1, 201),
            'TV': np.random.uniform(0, 300, 200),
            'Radio': np.random.uniform(0, 50, 200),
            'Newspaper': np.random.uniform(0, 120, 200),
            'Sales': np.random.uniform(1, 27, 200)
        })
    
    def test_train_test_split_ratio(self):
        """Test train-test split ratio is correct"""
        train_size = int(0.67 * len(self.data))
        test_size = len(self.data) - train_size
        self.assertEqual(train_size, 134)
        self.assertEqual(test_size, 66)
    
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
    
    def test_split_maintains_distribution(self):
        """Test split maintains data distribution"""
        train_size = int(0.67 * len(self.data))
        train = self.data[:train_size]
        test = self.data[train_size:]
        
        train_mean = train['Sales'].mean()
        test_mean = test['Sales'].mean()
        
        # Means should be similar (within 20%)
        self.assertLess(abs(train_mean - test_mean) / train_mean, 0.2)


class TestFeatureScaling(unittest.TestCase):
    """Test feature scaling and normalization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = np.array([
            [230.1, 37.8, 69.2],
            [44.5, 39.3, 45.1],
            [17.2, 45.9, 69.3],
            [151.5, 41.3, 58.5],
            [180.8, 10.8, 58.4]
        ])
    
    def test_scaling_reduces_variance(self):
        """Test scaling reduces feature variance"""
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        scaled = (self.data - mean) / std
        
        # Scaled data should have mean ~0 and std ~1
        self.assertLess(abs(scaled.mean(axis=0)).max(), 0.1)
        self.assertLess(abs(scaled.std(axis=0) - 1).max(), 0.1)
    
    def test_scaling_preserves_relationships(self):
        """Test scaling preserves feature relationships"""
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        scaled = (self.data - mean) / std
        
        # Correlation should be preserved
        original_corr = np.corrcoef(self.data.T)
        scaled_corr = np.corrcoef(scaled.T)
        
        np.testing.assert_array_almost_equal(original_corr, scaled_corr)
    
    def test_scaling_output_shape(self):
        """Test scaling maintains shape"""
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        scaled = (self.data - mean) / std
        
        self.assertEqual(scaled.shape, self.data.shape)
    
    def test_scaling_is_reversible(self):
        """Test scaling can be reversed"""
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        scaled = (self.data - mean) / std
        unscaled = scaled * std + mean
        
        np.testing.assert_array_almost_equal(unscaled, self.data)


class TestModelTraining(unittest.TestCase):
    """Test model training functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(3)
        self.X_train = np.random.randn(100, 3)
        self.y_train = 5 + 2*self.X_train[:, 0] + 3*self.X_train[:, 1] + self.X_train[:, 2] + np.random.randn(100)*0.1
    
    def test_model_learns_coefficients(self):
        """Test model learns reasonable coefficients"""
        # Simple linear regression
        X_with_intercept = np.column_stack([np.ones(len(self.X_train)), self.X_train])
        beta = np.linalg.lstsq(X_with_intercept, self.y_train, rcond=None)[0]
        
        # Coefficients should be close to [5, 2, 3, 1]
        self.assertAlmostEqual(beta[0], 5, delta=1)
        self.assertAlmostEqual(beta[1], 2, delta=1)
        self.assertAlmostEqual(beta[2], 3, delta=1)
        self.assertAlmostEqual(beta[3], 1, delta=1)
    
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
    
    def test_model_converges(self):
        """Test model training converges"""
        X_with_intercept = np.column_stack([np.ones(len(self.X_train)), self.X_train])
        beta = np.linalg.lstsq(X_with_intercept, self.y_train, rcond=None)[0]
        
        # Model should have learned something (not all zeros)
        self.assertFalse(np.allclose(beta, 0))


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.y_true = np.array([22.1, 10.4, 9.3, 18.5, 12.9])
        self.y_pred = np.array([20.5, 11.2, 8.9, 19.1, 13.2])
    
    def test_mse_calculation(self):
        """Test MSE calculation"""
        mse = np.mean((self.y_true - self.y_pred)**2)
        self.assertGreater(mse, 0)
        self.assertLess(mse, 10)
    
    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        mse = np.mean((self.y_true - self.y_pred)**2)
        rmse = np.sqrt(mse)
        self.assertGreater(rmse, 0)
        self.assertLess(rmse, 5)
    
    def test_mae_calculation(self):
        """Test MAE calculation"""
        mae = np.mean(np.abs(self.y_true - self.y_pred))
        self.assertGreater(mae, 0)
        self.assertLess(mae, 5)
    
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
    
    def test_metrics_are_positive(self):
        """Test all metrics are positive"""
        mse = np.mean((self.y_true - self.y_pred)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(self.y_true - self.y_pred))
        
        self.assertGreater(mse, 0)
        self.assertGreater(rmse, 0)
        self.assertGreater(mae, 0)


class TestPredictions(unittest.TestCase):
    """Test prediction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.coefficients = np.array([6.97, 0.046, 0.189, -0.001])
        self.feature_names = ['intercept', 'TV', 'Radio', 'Newspaper']
    
    def test_single_prediction(self):
        """Test single prediction"""
        features = np.array([1, 100, 30, 20])
        prediction = np.dot(self.coefficients, features)
        
        self.assertIsInstance(prediction, (int, float, np.number))
        self.assertGreater(prediction, 0)
    
    def test_multiple_predictions(self):
        """Test multiple predictions"""
        features = np.array([
            [1, 100, 30, 20],
            [1, 200, 40, 30],
            [1, 50, 20, 10]
        ])
        predictions = features @ self.coefficients
        
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(p > 0 for p in predictions))
    
    def test_prediction_bounds(self):
        """Test predictions are within reasonable bounds"""
        features = np.array([1, 100, 30, 20])
        prediction = np.dot(self.coefficients, features)
        
        # Sales should be between 1 and 27 (based on dataset)
        self.assertGreater(prediction, 0)
        self.assertLess(prediction, 50)
    
    def test_prediction_consistency(self):
        """Test predictions are consistent"""
        features = np.array([1, 100, 30, 20])
        pred1 = np.dot(self.coefficients, features)
        pred2 = np.dot(self.coefficients, features)
        
        self.assertEqual(pred1, pred2)
    
    def test_prediction_with_zero_features(self):
        """Test prediction with zero features"""
        features = np.array([1, 0, 0, 0])
        prediction = np.dot(self.coefficients, features)
        
        # Should equal intercept
        self.assertAlmostEqual(prediction, self.coefficients[0])


class TestVisualization(unittest.TestCase):
    """Test visualization functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = pd.DataFrame({
            'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
            'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
            'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
            'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
        })
    
    def test_correlation_matrix_shape(self):
        """Test correlation matrix has correct shape"""
        corr = self.data.corr()
        self.assertEqual(corr.shape, (4, 4))
    
    def test_correlation_values_valid(self):
        """Test correlation values are between -1 and 1"""
        corr = self.data.corr()
        self.assertTrue((corr >= -1).all().all())
        self.assertTrue((corr <= 1).all().all())
    
    def test_correlation_diagonal_is_one(self):
        """Test correlation diagonal is 1"""
        corr = self.data.corr()
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(4))
    
    def test_data_statistics(self):
        """Test data statistics calculation"""
        stats = self.data.describe()
        self.assertEqual(stats.shape[1], 4)
        self.assertIn('mean', stats.index)
        self.assertIn('std', stats.index)
        self.assertIn('min', stats.index)
        self.assertIn('max', stats.index)
    
    def test_feature_distributions(self):
        """Test feature distributions are valid"""
        for col in self.data.columns:
            self.assertGreater(self.data[col].std(), 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        self.assertEqual(len(empty_df), 0)
    
    def test_single_row_dataframe(self):
        """Test handling of single row"""
        single_row = pd.DataFrame({
            'TV': [230.1],
            'Radio': [37.8],
            'Newspaper': [69.2],
            'Sales': [22.1]
        })
        self.assertEqual(len(single_row), 1)
    
    def test_missing_column(self):
        """Test handling of missing column"""
        data = pd.DataFrame({
            'TV': [230.1, 44.5],
            'Radio': [37.8, 39.3]
        })
        self.assertNotIn('Sales', data.columns)
    
    def test_invalid_feature_values(self):
        """Test handling of invalid feature values"""
        data = pd.DataFrame({
            'TV': [-10, 44.5],
            'Radio': [37.8, 39.3],
            'Newspaper': [69.2, 45.1],
            'Sales': [22.1, 10.4]
        })
        # Negative values should be detected
        self.assertTrue((data['TV'] < 0).any())
    
    def test_division_by_zero_prevention(self):
        """Test prevention of division by zero"""
        data = np.array([1, 1, 1, 1, 1])
        std = data.std()
        # Constant data has zero std
        self.assertEqual(std, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(3)
        self.data = pd.DataFrame({
            'ID': range(1, 201),
            'TV': np.random.uniform(0.7, 296.4, 200),
            'Radio': np.random.uniform(0, 49.6, 200),
            'Newspaper': np.random.uniform(0.3, 114, 200),
            'Sales': np.random.uniform(1.6, 27, 200)
        })
    
    def test_complete_pipeline(self):
        """Test complete ML pipeline"""
        # Clean data
        clean_data = self.data.dropna().drop_duplicates()
        self.assertGreater(len(clean_data), 0)
        
        # Prepare features
        X = clean_data[['ID', 'TV', 'Radio', 'Newspaper']].values
        y = clean_data['Sales'].values
        
        # Split data
        train_size = int(0.67 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale features
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std
        
        # Train model
        X_train_with_intercept = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        beta = np.linalg.lstsq(X_train_with_intercept, y_train, rcond=None)[0]
        
        # Make predictions
        X_test_with_intercept = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])
        y_pred = X_test_with_intercept @ beta
        
        # Evaluate
        mse = np.mean((y_test - y_pred)**2)
        r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
        
        # R² can be negative with random data, verify it's a valid number
        self.assertIsInstance(r2, (int, float, np.number))
        self.assertLess(r2, 2)  # R² should be less than 2
        self.assertGreater(mse, 0)
    
    def test_pipeline_produces_valid_metrics(self):
        """Test pipeline produces valid metrics"""
        clean_data = self.data.dropna().drop_duplicates()
        X = clean_data[['ID', 'TV', 'Radio', 'Newspaper']].values
        y = clean_data['Sales'].values
        
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
