#!/usr/bin/env python3
"""
MACHINE LEARNING SYSTEM - TDD IMPLEMENTATION EXAMPLE
Demonstrates pure TDD-first development for machine learning
"""

import unittest
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

@dataclass
class MLModelResult:
    """Result structure for ML operations"""

    def __init__(self, success: bool, data: Any, message: str, metadata: Dict[str, Any] = None):
        self.success = success
        self.data = data
        self.message = message
        self.metadata = metadata or {}

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_time: float
    prediction_time: float

@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    num_samples: int
    num_features: int
    target_distribution: Dict[str, float]
    missing_values: int

class DataPreprocessingService:
    """
    Data preprocessing service
    Built using pure Test-Driven Development methodology
    """

    def __init__(self):
        """Initialize the data preprocessing service"""
        self.config = self._load_configuration()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration settings"""
        return {
            'max_missing_value_ratio': 0.1,
            'min_feature_importance': 0.01,
            'outlier_z_score_threshold': 3.0,
            'test_split_ratio': 0.2
        }

    def validate_input(self, input_data: Any) -> MLModelResult:
        """Validate input data"""
        if input_data is None:
            return MLModelResult(
                success=False,
                data=None,
                message="Input data cannot be None",
                metadata={'error_type': 'null_input'}
            )

        if not hasattr(input_data, '__len__'):
            return MLModelResult(
                success=False,
                data=None,
                message="Input must be array-like",
                metadata={'error_type': 'invalid_type'}
            )

        return MLModelResult(
            success=True,
            data=input_data,
            message="Input validation passed"
        )

    def load_data(self, data_path: str) -> MLModelResult:
        """Load data from file"""
        try:
            # For demo purposes, create synthetic data
            if 'iris' in data_path.lower():
                return self._load_iris_data()
            elif 'digits' in data_path.lower():
                return self._load_digits_data()
            else:
                return MLModelResult(
                    success=False,
                    data=None,
                    message=f"Dataset '{data_path}' not supported",
                    metadata={'error_type': 'unsupported_dataset'}
                )

        except Exception as e:
            return MLModelResult(
                success=False,
                data=None,
                message=f"Error loading data: {str(e)}",
                metadata={'error_type': 'data_loading_error'}
            )

    def _load_iris_data(self) -> MLModelResult:
        """Load Iris dataset"""
        from sklearn.datasets import load_iris

        iris = load_iris()
        X, y = iris.data, iris.target

        data = {
            'features': X,
            'targets': y,
            'feature_names': iris.feature_names,
            'target_names': iris.target_names.tolist()
        }

        return MLModelResult(
            success=True,
            data=data,
            message="Iris dataset loaded successfully",
            metadata={
                'dataset_name': 'iris',
                'num_samples': len(X),
                'num_features': X.shape[1]
            }
        )

    def _load_digits_data(self) -> MLModelResult:
        """Load Digits dataset"""
        from sklearn.datasets import load_digits

        digits = load_digits()
        X, y = digits.data, digits.target

        data = {
            'features': X,
            'targets': y,
            'feature_names': [f'pixel_{i}' for i in range(X.shape[1])],
            'target_names': [str(i) for i in range(10)]
        }

        return MLModelResult(
            success=True,
            data=data,
            message="Digits dataset loaded successfully",
            metadata={
                'dataset_name': 'digits',
                'num_samples': len(X),
                'num_features': X.shape[1]
            }
        )

    def preprocess_data(self, features: np.ndarray, targets: np.ndarray) -> MLModelResult:
        """Preprocess data for ML"""
        try:
            validation = self.validate_input(features)
            if not validation.success:
                return validation

            # Handle missing values
            features_clean = self._handle_missing_values(features)

            # Remove outliers
            features_clean = self._remove_outliers(features_clean)

            # Normalize features
            features_normalized = self._normalize_features(features_clean)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_normalized, targets,
                test_size=self.config['test_split_ratio'],
                random_state=42
            )

            preprocessed_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'original_shape': features.shape
            }

            return MLModelResult(
                success=True,
                data=preprocessed_data,
                message="Data preprocessing completed successfully",
                metadata={'preprocessing_steps': ['missing_values', 'outliers', 'normalization', 'splitting']}
            )

        except Exception as e:
            return MLModelResult(
                success=False,
                data=None,
                message=f"Error preprocessing data: {str(e)}",
                metadata={'error_type': 'preprocessing_error'}
            )

    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in features"""
        # For demo, assume no missing values
        return features

    def _remove_outliers(self, features: np.ndarray) -> np.ndarray:
        """Remove outliers using Z-score method"""
        # Simplified outlier removal
        z_scores = np.abs((features - features.mean(axis=0)) / features.std(axis=0))
        mask = z_scores < self.config['outlier_z_score_threshold']
        return features[mask.all(axis=1)]

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        return (features - min_vals) / (max_vals - min_vals)

    def get_data_info(self, data: Dict[str, Any]) -> DatasetInfo:
        """Get information about the dataset"""
        features = data['features']
        targets = data['targets']

        unique_targets, counts = np.unique(targets, return_counts=True)
        target_distribution = {
            str(int(target)): count / len(targets)
            for target, count in zip(unique_targets, counts)
        }

        return DatasetInfo(
            name=data.get('dataset_name', 'unknown'),
            num_samples=len(features),
            num_features=features.shape[1],
            target_distribution=target_distribution,
            missing_values=0  # Simplified for demo
        )

class ModelTrainingService:
    """
    Model training service
    Built using pure Test-Driven Development methodology
    """

    def __init__(self, preprocessing_service: DataPreprocessingService):
        """Initialize the model training service"""
        self.preprocessing_service = preprocessing_service
        self.models = {}
        self.model_counter = 0

    def train_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray) -> MLModelResult:
        """Train a machine learning model"""
        try:
            if model_type == 'logistic_regression':
                return self._train_logistic_regression(X_train, y_train)
            elif model_type == 'random_forest':
                return self._train_random_forest(X_train, y_train)
            elif model_type == 'svm':
                return self._train_svm(X_train, y_train)
            else:
                return MLModelResult(
                    success=False,
                    data=None,
                    message=f"Unsupported model type: {model_type}",
                    metadata={'error_type': 'unsupported_model'}
                )

        except Exception as e:
            return MLModelResult(
                success=False,
                data=None,
                message=f"Error training model: {str(e)}",
                metadata={'error_type': 'training_error'}
            )

    def _train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> MLModelResult:
        """Train logistic regression model"""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        self.model_counter += 1
        model_id = self.model_counter

        self.models[model_id] = {
            'model': model,
            'type': 'logistic_regression',
            'feature_count': X_train.shape[1]
        }

        return MLModelResult(
            success=True,
            data={'model_id': model_id, 'model_type': 'logistic_regression'},
            message="Logistic regression model trained successfully",
            metadata={'feature_count': X_train.shape[1], 'sample_count': len(X_train)}
        )

    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> MLModelResult:
        """Train random forest model"""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        self.model_counter += 1
        model_id = self.model_counter

        self.models[model_id] = {
            'model': model,
            'type': 'random_forest',
            'feature_count': X_train.shape[1]
        }

        return MLModelResult(
            success=True,
            data={'model_id': model_id, 'model_type': 'random_forest'},
            message="Random forest model trained successfully",
            metadata={'feature_count': X_train.shape[1], 'sample_count': len(X_train)}
        )

    def _train_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> MLModelResult:
        """Train SVM model"""
        from sklearn.svm import SVC

        model = SVC(random_state=42)
        model.fit(X_train, y_train)

        self.model_counter += 1
        model_id = self.model_counter

        self.models[model_id] = {
            'model': model,
            'type': 'svm',
            'feature_count': X_train.shape[1]
        }

        return MLModelResult(
            success=True,
            data={'model_id': model_id, 'model_type': 'svm'},
            message="SVM model trained successfully",
            metadata={'feature_count': X_train.shape[1], 'sample_count': len(X_train)}
        )

    def predict(self, model_id: int, X_test: np.ndarray) -> MLModelResult:
        """Make predictions using trained model"""
        try:
            if model_id not in self.models:
                return MLModelResult(
                    success=False,
                    data=None,
                    message=f"Model {model_id} not found",
                    metadata={'error_type': 'model_not_found'}
                )

            model_info = self.models[model_id]
            model = model_info['model']

            predictions = model.predict(X_test)

            return MLModelResult(
                success=True,
                data=predictions,
                message="Predictions generated successfully",
                metadata={
                    'model_id': model_id,
                    'model_type': model_info['type'],
                    'prediction_count': len(predictions)
                }
            )

        except Exception as e:
            return MLModelResult(
                success=False,
                data=None,
                message=f"Error making predictions: {str(e)}",
                metadata={'error_type': 'prediction_error'}
            )

class ModelEvaluationService:
    """
    Model evaluation service
    Built using pure Test-Driven Development methodology
    """

    def __init__(self, training_service: ModelTrainingService):
        """Initialize the model evaluation service"""
        self.training_service = training_service

    def evaluate_model(self, model_id: int, X_test: np.ndarray, y_test: np.ndarray) -> MLModelResult:
        """Evaluate model performance"""
        try:
            # Get predictions
            prediction_result = self.training_service.predict(model_id, X_test)
            if not prediction_result.success:
                return prediction_result

            predictions = prediction_result.data

            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')

            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                train_time=0.0,  # Would be tracked in real implementation
                prediction_time=0.0
            )

            model_info = self.training_service.models.get(model_id, {})
            model_type = model_info.get('type', 'unknown')

            return MLModelResult(
                success=True,
                data=metrics,
                message="Model evaluation completed successfully",
                metadata={
                    'model_id': model_id,
                    'model_type': model_type,
                    'test_samples': len(y_test)
                }
            )

        except Exception as e:
            return MLModelResult(
                success=False,
                data=None,
                message=f"Error evaluating model: {str(e)}",
                metadata={'error_type': 'evaluation_error'}
            )

    def generate_evaluation_report(self, model_id: int, metrics: ModelMetrics) -> str:
        """Generate evaluation report"""
        report = f"""
# Machine Learning Model Evaluation Report

## Model ID: {model_id}

### Performance Metrics
- **Accuracy**: {metrics.accuracy".4f"}
- **Precision**: {metrics.precision".4f"}
- **Recall**: {metrics.recall".4f"}
- **F1 Score**: {metrics.f1_score".4f"}

### Timing Information
- **Training Time**: {metrics.train_time".2f"} seconds
- **Prediction Time**: {metrics.prediction_time".2f"} seconds

### Assessment
"""
        if metrics.accuracy > 0.9:
            report += "✅ Excellent model performance\n"
        elif metrics.accuracy > 0.8:
            report += "✅ Good model performance\n"
        elif metrics.accuracy > 0.7:
            report += "⚠️  Moderate model performance\n"
        else:
            report += "❌ Poor model performance\n"

        return report

# Factory functions
def create_data_preprocessing_service() -> DataPreprocessingService:
    """Factory function for data preprocessing service"""
    return DataPreprocessingService()

def create_model_training_service(preprocessing_service: DataPreprocessingService) -> ModelTrainingService:
    """Factory function for model training service"""
    return ModelTrainingService(preprocessing_service)

def create_model_evaluation_service(training_service: ModelTrainingService) -> ModelEvaluationService:
    """Factory function for model evaluation service"""
    return ModelEvaluationService(training_service)

# Comprehensive test suite demonstrating TDD approach
class TestDataPreprocessingService(unittest.TestCase):
    """Test suite for data preprocessing service"""

    def setUp(self):
        """Set up test fixtures"""
        self.service = create_data_preprocessing_service()

    def tearDown(self):
        """Clean up after tests"""
        pass

    # Unit Tests
    def test_input_validation(self):
        """Test input validation"""
        # Valid inputs
        valid_inputs = [np.array([[1, 2], [3, 4]]), [1, 2, 3], (1, 2, 3)]
        for valid_input in valid_inputs:
            result = self.service.validate_input(valid_input)
            self.assertTrue(result.success, f"Input {type(valid_input)} should be valid")

        # Invalid inputs
        invalid_inputs = [None, "string", 123]
        for invalid_input in invalid_inputs:
            result = self.service.validate_input(invalid_input)
            self.assertFalse(result.success, f"Input {type(invalid_input)} should be invalid")

    def test_iris_data_loading(self):
        """Test Iris dataset loading"""
        result = self.service.load_data('iris')
        self.assertTrue(result.success)
        self.assertIn('features', result.data)
        self.assertIn('targets', result.data)
        self.assertEqual(len(result.data['features']), 150)  # Iris has 150 samples

    def test_digits_data_loading(self):
        """Test Digits dataset loading"""
        result = self.service.load_data('digits')
        self.assertTrue(result.success)
        self.assertIn('features', result.data)
        self.assertIn('targets', result.data)
        self.assertEqual(len(result.data['features']), 1797)  # Digits has 1797 samples

    def test_data_preprocessing(self):
        """Test data preprocessing"""
        # Create test data
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)

        result = self.service.preprocess_data(X, y)
        self.assertTrue(result.success)
        self.assertIn('X_train', result.data)
        self.assertIn('X_test', result.data)
        self.assertIn('y_train', result.data)
        self.assertIn('y_test', result.data)

        # Check train/test split
        total_samples = len(X)
        train_size = len(result.data['X_train'])
        test_size = len(result.data['X_test'])
        self.assertEqual(train_size + test_size, total_samples)

    # Edge Case Tests
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        X_empty = np.array([])
        y_empty = np.array([])

        result = self.service.preprocess_data(X_empty, y_empty)
        self.assertFalse(result.success)

    def test_single_sample_handling(self):
        """Test handling of single sample"""
        X_single = np.array([[1, 2, 3, 4]])
        y_single = np.array([0])

        result = self.service.preprocess_data(X_single, y_single)
        self.assertTrue(result.success)

class TestModelTrainingService(unittest.TestCase):
    """Test suite for model training service"""

    def setUp(self):
        """Set up test fixtures"""
        self.preprocessing_service = create_data_preprocessing_service()
        self.training_service = create_model_training_service(self.preprocessing_service)

    def tearDown(self):
        """Clean up after tests"""
        pass

    # Unit Tests
    def test_logistic_regression_training(self):
        """Test logistic regression training"""
        # Load and preprocess data
        data_result = self.preprocessing_service.load_data('iris')
        preprocess_result = self.preprocessing_service.preprocess_data(
            data_result.data['features'], data_result.data['targets']
        )

        X_train = preprocess_result.data['X_train']
        y_train = preprocess_result.data['y_train']

        # Train model
        result = self.training_service.train_model('logistic_regression', X_train, y_train)
        self.assertTrue(result.success)
        self.assertIn('model_id', result.data)

    def test_random_forest_training(self):
        """Test random forest training"""
        # Load and preprocess data
        data_result = self.preprocessing_service.load_data('iris')
        preprocess_result = self.preprocessing_service.preprocess_data(
            data_result.data['features'], data_result.data['targets']
        )

        X_train = preprocess_result.data['X_train']
        y_train = preprocess_result.data['y_train']

        # Train model
        result = self.training_service.train_model('random_forest', X_train, y_train)
        self.assertTrue(result.success)
        self.assertIn('model_id', result.data)

    def test_svm_training(self):
        """Test SVM training"""
        # Load and preprocess data
        data_result = self.preprocessing_service.load_data('iris')
        preprocess_result = self.preprocessing_service.preprocess_data(
            data_result.data['features'], data_result.data['targets']
        )

        X_train = preprocess_result.data['X_train']
        y_train = preprocess_result.data['y_train']

        # Train model
        result = self.training_service.train_model('svm', X_train, y_train)
        self.assertTrue(result.success)
        self.assertIn('model_id', result.data)

    def test_unsupported_model_type(self):
        """Test handling of unsupported model types"""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        result = self.training_service.train_model('unsupported_model', X, y)
        self.assertFalse(result.success)
        self.assertIn('Unsupported model type', result.message)

    # Integration Tests
    def test_complete_training_workflow(self):
        """Test complete model training workflow"""
        # Load data
        data_result = self.preprocessing_service.load_data('iris')
        self.assertTrue(data_result.success)

        # Preprocess data
        preprocess_result = self.preprocessing_service.preprocess_data(
            data_result.data['features'], data_result.data['targets']
        )
        self.assertTrue(preprocess_result.success)

        # Train multiple models
        models = ['logistic_regression', 'random_forest', 'svm']
        trained_models = []

        for model_type in models:
            result = self.training_service.train_model(
                model_type,
                preprocess_result.data['X_train'],
                preprocess_result.data['y_train']
            )
            self.assertTrue(result.success)
            trained_models.append(result.data['model_id'])

        # Should have trained 3 models
        self.assertEqual(len(trained_models), 3)

class TestModelEvaluationService(unittest.TestCase):
    """Test suite for model evaluation service"""

    def setUp(self):
        """Set up test fixtures"""
        self.preprocessing_service = create_data_preprocessing_service()
        self.training_service = create_model_training_service(self.preprocessing_service)
        self.evaluation_service = create_model_evaluation_service(self.training_service)

    def tearDown(self):
        """Clean up after tests"""
        pass

    # Unit Tests
    def test_model_evaluation(self):
        """Test model evaluation"""
        # Load and preprocess data
        data_result = self.preprocessing_service.load_data('iris')
        preprocess_result = self.preprocessing_service.preprocess_data(
            data_result.data['features'], data_result.data['targets']
        )

        # Train model
        train_result = self.training_service.train_model(
            'logistic_regression',
            preprocess_result.data['X_train'],
            preprocess_result.data['y_train']
        )
        model_id = train_result.data['model_id']

        # Evaluate model
        result = self.evaluation_service.evaluate_model(
            model_id,
            preprocess_result.data['X_test'],
            preprocess_result.data['y_test']
        )

        self.assertTrue(result.success)
        self.assertIsInstance(result.data, ModelMetrics)
        self.assertGreater(result.data.accuracy, 0.0)
        self.assertLessEqual(result.data.accuracy, 1.0)

    def test_evaluation_report_generation(self):
        """Test evaluation report generation"""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.95,
            f1_score=0.945,
            train_time=0.5,
            prediction_time=0.05
        )

        report = self.evaluation_service.generate_evaluation_report(1, metrics)

        self.assertIn('Model ID: 1', report)
        self.assertIn('Accuracy: 0.9500', report)
        self.assertIn('Excellent model performance', report)

    # Integration Tests
    def test_complete_ml_workflow(self):
        """Test complete ML workflow from data loading to evaluation"""
        # Load data
        data_result = self.preprocessing_service.load_data('iris')
        self.assertTrue(data_result.success)

        # Preprocess data
        preprocess_result = self.preprocessing_service.preprocess_data(
            data_result.data['features'], data_result.data['targets']
        )
        self.assertTrue(preprocess_result.success)

        # Train model
        train_result = self.training_service.train_model(
            'random_forest',
            preprocess_result.data['X_train'],
            preprocess_result.data['y_train']
        )
        self.assertTrue(train_result.success)
        model_id = train_result.data['model_id']

        # Evaluate model
        eval_result = self.evaluation_service.evaluate_model(
            model_id,
            preprocess_result.data['X_test'],
            preprocess_result.data['y_test']
        )
        self.assertTrue(eval_result.success)

        # Check metrics
        metrics = eval_result.data
        self.assertGreater(metrics.accuracy, 0.8)  # Should be reasonably accurate
        self.assertGreater(metrics.f1_score, 0.8)

    def test_model_comparison(self):
        """Test comparison of different models"""
        # Load and preprocess data
        data_result = self.preprocessing_service.load_data('iris')
        preprocess_result = self.preprocessing_service.preprocess_data(
            data_result.data['features'], data_result.data['targets']
        )

        # Train multiple models
        models = ['logistic_regression', 'random_forest', 'svm']
        model_results = {}

        for model_type in models:
            train_result = self.training_service.train_model(
                model_type,
                preprocess_result.data['X_train'],
                preprocess_result.data['y_train']
            )
            self.assertTrue(train_result.success)

            eval_result = self.evaluation_service.evaluate_model(
                train_result.data['model_id'],
                preprocess_result.data['X_test'],
                preprocess_result.data['y_test']
            )
            self.assertTrue(eval_result.success)

            model_results[model_type] = eval_result.data

        # All models should have reasonable performance
        for model_type, metrics in model_results.items():
            self.assertGreater(metrics.accuracy, 0.7, f"{model_type} should have reasonable accuracy")

if __name__ == "__main__":
    # Run comprehensive test suite
    unittest.main(verbosity=2)