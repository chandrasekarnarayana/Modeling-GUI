import unittest
import numpy as np
from modeling_gui.models import ModelManager

class TestModelManager(unittest.TestCase):
    
    def setUp(self):
        """Set up the test environment with sample data."""
        self.X = np.array([[1], [2], [3], [4], [5]])  # Simple input data (features)
        self.Y = np.array([2, 4, 6, 8, 10])          # Output data (target)
        self.model_manager = ModelManager()           # Create an instance of ModelManager
    
    def test_ols_model(self):
        """Test the Ordinary Least Squares (OLS) model."""
        model = self.model_manager.ols(self.X, self.Y)
        predictions = model.predict(np.array([[1], [2], [3], [4], [5]]))
        self.assertTrue(np.allclose(predictions, self.Y), "OLS predictions are incorrect.")

    def test_random_forest_regression(self):
        """Test the Random Forest model for regression."""
        model = self.model_manager.random_forest(self.X, self.Y, n_estimators=10, max_depth=5)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.Y), "Random Forest regression failed to return correct number of predictions.")

    def test_gaussian_fit(self):
        """Test Gaussian fitting on sample data."""
        mean, std_dev, gaussian = self.model_manager.gaussian_fit(self.X, self.Y)
        self.assertIsNotNone(gaussian, "Gaussian fitting failed.")
        self.assertGreater(mean, 0, "Gaussian mean is incorrect.")
        self.assertGreater(std_dev, 0, "Gaussian std_dev is incorrect.")

    def test_exponential_fit(self):
        """Test exponential fitting."""
        params = self.model_manager.exponential_fit(self.X.flatten(), self.Y)
        self.assertEqual(len(params), 3, "Exponential fit should return 3 parameters.")
        self.assertGreater(params[0], 0, "Exponential fit parameter 'a' is incorrect.")
    
    def tearDown(self):
        """Clean up any necessary data after tests."""
        pass

if __name__ == '__main__':
    unittest.main()

