"""
MLE-STAR Agent for MLE-Dojo-gepa
Machine Learning Engineering - Systematic Training and Refinement
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MLEStarEnsemble(BaseEstimator):
    """
    MLE-STAR: Machine Learning Engineering - Systematic Training and Refinement
    Advanced ensemble learning agent with automatic model selection and optimization.
    """
    
    def __init__(self, 
                 base_estimators: Optional[List[BaseEstimator]] = None,
                 task_type: str = 'classification',
                 n_models: int = 5,
                 random_state: int = 42):
        """
        Initialize the MLE-STAR ensemble agent.
        
        Args:
            base_estimators: List of base estimators to use in the ensemble.
                           If None, will create default estimators based on task_type.
            task_type: Type of task - 'classification' or 'regression'.
            n_models: Number of models to include in the ensemble.
            random_state: Random seed for reproducibility.
        """
        self.base_estimators = base_estimators
        self.task_type = task_type
        self.n_models = n_models
        self.random_state = random_state
        self.weights_ = None
        self.fitted_models_ = []
        self.feature_importances_ = None
        
        if self.base_estimators is None:
            self._initialize_default_estimators()
            
        np.random.seed(self.random_state)
    
    def _initialize_default_estimators(self):
        """Initialize default base estimators based on task type."""
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor
        )
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.svm import SVC, SVR
        from xgboost import XGBClassifier, XGBRegressor
        from lightgbm import LGBMClassifier, LGBMRegressor
        
        if self.task_type == 'classification':
            self.base_estimators = [
                RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
                LogisticRegression(max_iter=1000, random_state=self.random_state),
                SVC(probability=True, random_state=self.random_state),
                XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                            random_state=self.random_state),
                LGBMClassifier(random_state=self.random_state)
            ]
        else:  # regression
            self.base_estimators = [
                RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
                Ridge(random_state=self.random_state),
                SVR(),
                XGBRegressor(random_state=self.random_state),
                LGBMRegressor(random_state=self.random_state)
            ]
        
        # Limit number of models if needed
        if self.n_models < len(self.base_estimators):
            np.random.shuffle(self.base_estimators)
            self.base_estimators = self.base_estimators[:self.n_models]
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the MLE-STAR ensemble on the training data.
        
        Args:
            X: Training data features
            y: Training data target
            sample_weight: Sample weights for training
            
        Returns:
            self: Returns self.
        """
        # Fit all base models
        self.fitted_models_ = []
        for model in self.base_estimators:
            model_clone = clone(model)
            if sample_weight is not None:
                model_clone.fit(X, y, sample_weight=sample_weight)
            else:
                model_clone.fit(X, y)
            self.fitted_models_.append(model_clone)
        
        # Calculate model weights based on cross-validated performance
        self._calculate_weights(X, y)
        
        # Calculate feature importances
        self._calculate_feature_importances(X)
        
        return self
    
    def _calculate_weights(self, X, y):
        """Calculate weights for each model based on cross-validated performance."""
        n_samples = X.shape[0]
        n_folds = min(5, n_samples // 10)  # Use 5-fold or fewer if not enough samples
        n_folds = max(2, n_folds)  # At least 2 folds
        
        scores = []
        
        for model in self.fitted_models_:
            try:
                if self.task_type == 'classification':
                    preds = cross_val_predict(model, X, y, cv=n_folds, method='predict_proba')
                    # Use negative log loss as score (higher is better)
                    score = -log_loss(y, preds, normalize=True)
                else:  # regression
                    preds = cross_val_predict(model, X, y, cv=n_folds)
                    # Use negative MSE as score (higher is better)
                    score = -mean_squared_error(y, preds)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Error in cross-validation for {model.__class__.__name__}: {str(e)}")
                scores.append(-np.inf)  # Assign lowest weight if model fails
        
        # Convert scores to weights using softmax
        scores = np.array(scores)
        # Shift scores for numerical stability
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        self.weights_ = exp_scores / np.sum(exp_scores)
    
    def _calculate_feature_importances(self, X):
        """Calculate weighted feature importances across all models."""
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        for model, weight in zip(self.fitted_models_, self.weights_):
            try:
                if hasattr(model, 'feature_importances_'):
                    importances += weight * model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # For linear models, take absolute coefficients
                    coef = model.coef_
                    if len(coef.shape) > 1:  # For multi-class
                        coef = np.mean(np.abs(coef), axis=0)
                    importances += weight * coef
                else:
                    # For models without feature importances, use equal weights
                    importances += weight * np.ones(n_features) / n_features
            except Exception as e:
                print(f"Warning: Could not calculate feature importances for {model.__class__.__name__}: {str(e)}")
        
        # Normalize importances to sum to 1
        if np.sum(importances) > 0:
            self.feature_importances_ = importances / np.sum(importances)
        else:
            self.feature_importances_ = np.ones(n_features) / n_features
    
    def predict(self, X):
        """
        Predict using the MLE-STAR ensemble.
        
        Args:
            X: Input features
            
        Returns:
            array: Predicted values
        """
        if self.task_type == 'classification':
            return self._predict_classification(X)
        else:
            return self._predict_regression(X)
    
    def _predict_classification(self, X):
        """Predict class labels for classification."""
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.fitted_models_])
        
        # Weighted voting
        unique_labels = np.unique(np.concatenate([model.classes_ for model in self.fitted_models_ 
                                                if hasattr(model, 'classes_')]))
        
        if len(unique_labels) == 0:
            raise ValueError("No valid class labels found in the ensemble models.")
        
        weighted_votes = np.zeros((X.shape[0], len(unique_labels)))
        
        for i, model in enumerate(self.fitted_models_):
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                # Map class probabilities to the full set of unique labels
                for j, label in enumerate(model.classes_):
                    col_idx = np.where(unique_labels == label)[0][0]
                    weighted_votes[:, col_idx] += self.weights_[i] * probas[:, j]
            else:
                # For models without predict_proba, use one-hot encoding
                preds = model.predict(X)
                for j, label in enumerate(unique_labels):
                    mask = (preds == label).astype(int)
                    weighted_votes[:, j] += self.weights_[i] * mask
        
        # Return class with highest weighted probability
        return unique_labels[np.argmax(weighted_votes, axis=1)]
    
    def _predict_regression(self, X):
        """Predict values for regression."""
        predictions = np.array([model.predict(X) for model in self.fitted_models_])
        return np.average(predictions, axis=0, weights=self.weights_)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification.
        
        Args:
            X: Input features
            
        Returns:
            array: Predicted probabilities for each class
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks.")
        
        # Get probabilities from all models that support it
        probas = []
        valid_weights = []
        
        for model, weight in zip(self.fitted_models_, self.weights_):
            if hasattr(model, 'predict_proba'):
                probas.append(model.predict_proba(X))
                valid_weights.append(weight)
        
        if not probas:
            raise ValueError("None of the base estimators implement predict_proba")
        
        # Normalize weights
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / np.sum(valid_weights)
        
        # Weighted average of probabilities
        weighted_proba = np.zeros_like(probas[0])
        for proba, weight in zip(probas, valid_weights):
            # Ensure proba has the same number of columns as weighted_proba
            if proba.shape[1] < weighted_proba.shape[1]:
                # Handle case where some models don't predict all classes
                temp = np.zeros((proba.shape[0], weighted_proba.shape[1]))
                for i, cls in enumerate(model.classes_):
                    if cls in range(weighted_proba.shape[1]):
                        temp[:, cls] = proba[:, i]
                proba = temp
            weighted_proba += weight * proba
        
        return weighted_proba
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'base_estimators': self.base_estimators,
            'task_type': self.task_type,
            'n_models': self.n_models,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for param, value in params.items():
            setattr(self, param, value)
        return self


def test_mle_star_agent():
    """Test the MLE-STAR agent on sample datasets."""
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    print("Testing MLE-STAR Agent")
    print("=" * 50)
    
    # Test classification
    print("\nTesting Classification...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = MLEStarEnsemble(task_type='classification', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    # Test regression
    print("\nTesting Regression...")
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, 
                          noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    reg = MLEStarEnsemble(task_type='regression', random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Regression MSE: {mse:.4f}")
    
    # Show feature importances
    print("\nFeature Importances (Top 5):")
    top_5 = np.argsort(reg.feature_importances_)[::-1][:5]
    for i, idx in enumerate(top_5):
        print(f"  Feature {idx}: {reg.feature_importances_[idx]:.4f}")


if __name__ == "__main__":
    test_mle_star_agent()
