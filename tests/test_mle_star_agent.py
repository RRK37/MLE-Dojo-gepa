"""
Tests for MLE-STAR Agent
"""

import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.exceptions import NotFittedError

from mle_dojo_gepa.mle_star_agent import MLEStarEnsemble


def test_initialization():
    """Test MLE-STAR initialization with different parameters."""
    # Test default initialization
    ensemble = MLEStarEnsemble()
    assert ensemble.task_type == 'classification'
    assert ensemble.n_models == 5
    assert ensemble.random_state == 42
    
    # Test custom initialization
    ensemble = MLEStarEnsemble(task_type='regression', n_models=3, random_state=123)
    assert ensemble.task_type == 'regression'
    assert ensemble.n_models == 3
    assert ensemble.random_state == 123


def test_classification():
    """Test MLE-STAR on a classification task."""
    # Create synthetic classification data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Initialize and fit the ensemble
    ensemble = MLEStarEnsemble(
        task_type='classification',
        n_models=4,
        random_state=42
    )
    
    # Test before fitting
    with pytest.raises(NotFittedError):
        ensemble.predict(X_test)
    
    # Fit the model
    ensemble.fit(X_train, y_train)
    
    # Test predictions
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    
    # Check predictions shape and type
    assert y_pred.shape == (X_test.shape[0],)
    assert y_proba.shape == (X_test.shape[0], 3)  # 3 classes
    
    # Check probabilities sum to 1
    assert np.allclose(y_proba.sum(axis=1), 1.0)
    
    # Check accuracy is reasonable
    accuracy = accuracy_score(y_test, y_pred)
    assert 0.7 <= accuracy <= 1.0, f"Unexpected accuracy: {accuracy}"
    
    # Check feature importances
    assert hasattr(ensemble, 'feature_importances_')
    assert ensemble.feature_importances_.shape == (X.shape[1],)
    assert np.all(ensemble.feature_importances_ >= 0)
    assert np.isclose(np.sum(ensemble.feature_importances_), 1.0)


def test_regression():
    """Test MLE-STAR on a regression task."""
    # Create synthetic regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Initialize and fit the ensemble
    ensemble = MLEStarEnsemble(
        task_type='regression',
        n_models=4,
        random_state=42
    )
    
    # Fit the model
    ensemble.fit(X_train, y_train)
    
    # Test predictions
    y_pred = ensemble.predict(X_test)
    
    # Check predictions shape and type
    assert y_pred.shape == (X_test.shape[0],)
    
    # Check MSE is reasonable (should be close to noise level)
    mse = mean_squared_error(y_test, y_pred)
    assert 0 <= mse <= 0.2, f"Unexpected MSE: {mse}"
    
    # Check feature importances
    assert hasattr(ensemble, 'feature_importances_')
    assert ensemble.feature_importances_.shape == (X.shape[1],)
    assert np.all(ensemble.feature_importances_ >= 0)
    assert np.isclose(np.sum(ensemble.feature_importances_), 1.0)


def test_custom_estimators():
    """Test MLE-STAR with custom base estimators."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    
    # Create custom estimators
    estimators = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    # Initialize with custom estimators
    ensemble = MLEStarEnsemble(
        base_estimators=estimators,
        task_type='classification',
        random_state=42
    )
    
    # Create synthetic data
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    # Fit and predict
    ensemble.fit(X, y)
    y_pred = ensemble.predict(X)
    
    # Check predictions
    assert y_pred.shape == (X.shape[0],)
    assert accuracy_score(y, y_pred) > 0.8  # Should perform well on training data


def test_parallel_processing():
    """Test that the ensemble can be used with joblib parallel processing."""
    from joblib import Parallel, delayed
    
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    
    # Create multiple ensembles in parallel
    def train_ensemble(seed):
        ensemble = MLEStarEnsemble(
            task_type='regression',
            n_models=3,
            random_state=seed
        )
        ensemble.fit(X, y)
        return ensemble.predict(X[:5])
    
    # Run in parallel
    results = Parallel(n_jobs=2)(
        delayed(train_ensemble)(i) for i in range(3)
    )
    
    # Check results
    assert len(results) == 3
    for preds in results:
        assert preds.shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
