"""
BaseEstimator: The Foundation of Every ML Algorithm

This is the CONTRACT that every ML model must follow. Understanding this deeply
will make you write better ML code than 90% of engineers.

Key Papers/Concepts:
- scikit-learn API design paper: https://arxiv.org/abs/1309.0238
- "API design for machine learning software: experiences from the scikit-learn project"
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pickle
import json


class BaseEstimator(ABC):
    """
    The fundamental contract for all estimators in MiniKit.
    
    DESIGN PHILOSOPHY (from sklearn):
    1. Consistency - All objects share a common interface
    2. Inspection - Constructor params accessible as public attributes
    3. Non-proliferation of classes - Use composition over inheritance
    4. Sensible defaults - Models should work out-of-the-box
    5. Immutable constructor params - Set once, never change
    
    THE GOTCHA MOST PEOPLE MISS:
    fit() must ALWAYS return self. This enables method chaining:
    model.fit(X, y).predict(X_test)
    
    This pattern is called the "Builder Pattern" and it's why sklearn feels so clean.
    """
    
    def __init__(self):
        """
        CRITICAL PATTERN: Store ALL constructor params as public attributes.
        This enables get_params() and set_params() to work automatically.
        
        Example:
            def __init__(self, learning_rate=0.01, max_iter=100):
                self.learning_rate = learning_rate  # Store as-is
                self.max_iter = max_iter            # No transformation
        
        WHY: Enables cloning, grid search, and introspection.
        """
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Learn from data. This is where the magic happens.
        
        MUST RETURN self for method chaining!
        
        Args:
            X: Training data, shape (n_samples, n_features)
            y: Target values, shape (n_samples,) or (n_samples, n_outputs)
            **kwargs: Additional fit parameters (sample_weight, etc.)
        
        Returns:
            self: The fitted estimator
        
        PATTERN TO FOLLOW:
            def fit(self, X, y):
                # 1. Validate inputs
                X, y = self._validate_data(X, y)
                
                # 2. Initialize/reset internal state
                self._reset()
                
                # 3. Store training metadata (use trailing underscore)
                self.n_features_in_ = X.shape[1]
                self.classes_ = np.unique(y)  # For classifiers
                
                # 4. Actual learning algorithm
                self._fit_algorithm(X, y)
                
                # 5. Mark as fitted
                self._is_fitted = True
                
                # 6. MUST return self
                return self
        
        THE TRAILING UNDERSCORE CONVENTION:
        Attributes learned during fit() end with underscore:
        - self.coef_       (learned)
        - self.intercept_  (learned)
        vs constructor params:
        - self.learning_rate  (set by user)
        - self.max_iter       (set by user)
        
        This is NOT just convention - it's how you tell what's a hyperparameter
        vs what's learned from data.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        MUST call _check_is_fitted() first!
        
        Args:
            X: Test data, shape (n_samples, n_features)
        
        Returns:
            predictions: shape (n_samples,) or (n_samples, n_outputs)
        
        PATTERN:
            def predict(self, X):
                # 1. Check if fitted
                self._check_is_fitted()
                
                # 2. Validate input
                X = self._validate_data(X, reset=False)
                
                # 3. Make predictions
                return self._predict_algorithm(X)
        """
        pass
    
    def _check_is_fitted(self) -> None:
        """
        Ensure model has been fitted before prediction.
        
        THE GOTCHA: Python's duck typing means we could accidentally
        predict with uninitialized weights. This catches it early.
        
        IMPLEMENTATION OPTIONS:
        1. Simple flag (what we use):
           if not hasattr(self, '_is_fitted'):
               raise NotFittedError()
        
        2. Check for learned attributes:
           if not hasattr(self, 'coef_'):
               raise NotFittedError()
        
        3. sklearn's sophisticated version:
           check_is_fitted(self, ['coef_', 'intercept_'])
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this estimator."
            )
    
    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                       reset: bool = True) -> Union[np.ndarray, tuple]:
        """
        Input validation and conversion. This prevents 99% of bugs.
        
        Args:
            X: Input data
            y: Target data (optional)
            reset: If True, this is the first call (during fit)
                   If False, we're in predict/transform
        
        THE CRITICAL CHECKS:
        1. Convert to numpy if needed (pandas, lists, etc.)
        2. Check for NaN/Inf
        3. Ensure 2D shape for X
        4. Check consistent n_features between fit and predict
        5. Check X and y have same n_samples
        
        WHY reset PARAMETER:
        During fit(): reset=True, we STORE n_features_in_
        During predict(): reset=False, we CHECK against stored n_features_in_
        """
        # Convert to numpy
        X = self._ensure_numpy(X)
        
        if y is not None:
            y = self._ensure_numpy(y)
        
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        
        # Check for NaN/Inf
        if not np.isfinite(X).all():
            raise ValueError("Input contains NaN or infinity")
        
        if reset:
            # First call (fit) - store metadata
            self.n_features_in_ = X.shape[1]
        else:
            # Subsequent call (predict) - validate consistency
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                    f"was fitted with {self.n_features_in_} features"
                )
        
        # Validate y if provided
        if y is not None:
            if len(X) != len(y):
                raise ValueError(
                    f"X and y have inconsistent numbers of samples: "
                    f"{len(X)} != {len(y)}"
                )
            
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()  # Convert (n, 1) to (n,)
            
            return X, y
        
        return X
    
    @staticmethod
    def _ensure_numpy(data: Any) -> np.ndarray:
        """
        Convert various input types to numpy arrays.
        
        HANDLES:
        - Lists/tuples
        - Pandas DataFrame/Series
        - Numpy arrays (passthrough)
        - PyTorch tensors (if available)
        
        THE GOTCHA: pandas DataFrame.values vs .to_numpy()
        .values can return a view (dangerous!)
        .to_numpy() always returns a copy (safe but slower)
        """
        if isinstance(data, np.ndarray):
            return data
        
        # Handle pandas
        if hasattr(data, 'values'):  # DataFrame or Series
            return data.values
        
        # Handle PyTorch tensors
        if hasattr(data, 'numpy'):  # PyTorch tensor
            return data.detach().cpu().numpy()
        
        # Handle lists/tuples
        return np.asarray(data)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        THIS IS THE MAGIC that enables:
        - GridSearchCV
        - RandomizedSearchCV
        - Model cloning
        - Hyperparameter optimization
        
        PATTERN: Extract all constructor parameters automatically.
        
        Args:
            deep: If True, recursively get params for nested estimators
                  (e.g., Pipeline or VotingClassifier)
        
        THE IMPLEMENTATION TRICK:
        We use __init__ signature to find parameters, NOT __dict__.
        Why? Because __dict__ includes LEARNED attributes (coef_, etc.)
        which we don't want.
        
        Example:
            model = LogisticRegression(C=1.0, max_iter=100)
            model.fit(X, y)
            
            model.get_params()
            # {'C': 1.0, 'max_iter': 100}  # Only constructor params
            
            model.__dict__
            # {'C': 1.0, 'max_iter': 100, 'coef_': [...], '_is_fitted': True}
        """
        import inspect
        
        # Get constructor signature
        init_signature = inspect.signature(self.__init__)
        
        # Extract parameter names (excluding 'self')
        params = {}
        for param_name in init_signature.parameters:
            if param_name == 'self':
                continue
            
            # Get value from instance
            if hasattr(self, param_name):
                value = getattr(self, param_name)
                
                # Recursively get params for nested estimators
                if deep and hasattr(value, 'get_params'):
                    nested_params = value.get_params(deep=True)
                    for nested_key, nested_value in nested_params.items():
                        params[f"{param_name}__{nested_key}"] = nested_value
                
                params[param_name] = value
        
        return params
    
    def set_params(self, **params) -> 'BaseEstimator':
        """
        Set parameters for this estimator.
        
        THIS ENABLES:
        - GridSearchCV to test different hyperparameters
        - Pipeline to set nested parameters
        - Dynamic hyperparameter tuning
        
        MUST return self for method chaining!
        
        Args:
            **params: Estimator parameters
        
        Example:
            model = LogisticRegression()
            model.set_params(C=0.5, max_iter=200)
            model.fit(X, y)
        
        THE NESTED ESTIMATOR TRICK:
        You can set params in nested estimators using double underscore:
            pipeline = Pipeline([('scaler', StandardScaler()), 
                                ('clf', LogisticRegression())])
            pipeline.set_params(clf__C=0.5)  # Note the double underscore
        """
        if not params:
            return self
        
        valid_params = self.get_params(deep=True)
        
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {list(valid_params.keys())}"
                )
            
            # Handle nested parameters (contains __)
            if '__' in key:
                # e.g., 'clf__C' -> nested estimator
                parts = key.split('__')
                nested_estimator = getattr(self, parts[0])
                nested_estimator.set_params(**{parts[1]: value})
            else:
                # Direct parameter
                setattr(self, key, value)
        
        return self
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Default scoring method. Override in subclasses.
        
        For regressors: R² score
        For classifiers: Accuracy
        
        THIS IS VIRTUAL - it's meant to be overridden by mixins:
        - RegressorMixin.score() -> R²
        - ClassifierMixin.score() -> Accuracy
        
        THE PATTERN: This is the Template Method pattern.
        Base class defines the interface, subclasses provide implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement score(). "
            f"Use RegressorMixin or ClassifierMixin."
        )
    
    def __repr__(self) -> str:
        """
        Clean string representation showing all hyperparameters.
        
        Example output:
            LogisticRegression(C=1.0, max_iter=100, penalty='l2')
        
        THE SKLEARN PATTERN: Show constructor signature with current values.
        """
        params = self.get_params(deep=False)
        param_str = ', '.join([f"{k}={repr(v)}" for k, v in params.items()])
        return f"{self.__class__.__name__}({param_str})"
    
    def clone(self) -> 'BaseEstimator':
        """
        Create an unfitted copy of this estimator.
        
        CRITICAL FOR:
        - Cross-validation (each fold needs fresh model)
        - Ensemble methods (each base estimator is independent)
        - Hyperparameter search (avoid state leakage)
        
        THE IMPLEMENTATION:
        1. Get constructor params
        2. Create new instance with same params
        3. Return unfitted estimator
        
        WHY NOT deepcopy?
        deepcopy would copy learned attributes (coef_, etc.) too.
        We want a clean slate.
        """
        params = self.get_params(deep=False)
        return self.__class__(**params)
    
    def save(self, filepath: str) -> None:
        """
        Serialize model to disk.
        
        OPTIONS:
        1. pickle (Python-specific, full object graph)
        2. joblib (optimized for numpy arrays)
        3. Custom JSON (hyperparams only, then reconstruct)
        
        GOTCHA: pickle is NOT secure! Never unpickle untrusted data.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseEstimator':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _more_tags(self) -> Dict[str, Any]:
        """
        Metadata about the estimator's capabilities.
        
        SKLEARN USES THIS FOR:
        - Test suite (which tests to run)
        - Input validation (what types are supported)
        - Compatibility checks
        
        Example tags:
        {
            'requires_positive_X': False,
            'requires_positive_y': False,
            'binary_only': False,
            'multilabel': False,
            'multioutput': False,
            'multioutput_only': False,
            'no_validation': False,
            'poor_score': False,
            '_skip_test': False,
            'non_deterministic': False,
            'requires_fit': True,
        }
        """
        return {}


# =============================================================================
# MIXINS: The Power of Composition
# =============================================================================

class RegressorMixin:
    """
    Mixin class for all regression estimators.
    
    THE MIXIN PATTERN:
    Instead of deep inheritance hierarchies, we compose behavior.
    
    Usage:
        class LinearRegression(BaseEstimator, RegressorMixin):
            pass
    
    Now LinearRegression automatically gets:
    - score() method that computes R²
    - _estimator_type = 'regressor' tag
    """
    
    _estimator_type = "regressor"
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              sample_weight: Optional[np.ndarray] = None) -> float:
        """
        R² (coefficient of determination) score.
        
        FORMULA: R² = 1 - (SS_res / SS_tot)
        
        WHERE:
        - SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
        - SS_tot = Σ(y_true - y_mean)²  (total sum of squares)
        
        INTERPRETATION:
        - R² = 1.0: Perfect predictions
        - R² = 0.0: Model is as good as predicting mean
        - R² < 0.0: Model is WORSE than predicting mean (BAD!)
        
        THE GOTCHA: R² can be negative!
        This happens when your model is terrible.
        """
        y_pred = self.predict(X)
        
        if sample_weight is not None:
            # Weighted R²
            numerator = np.sum(sample_weight * (y - y_pred) ** 2)
            denominator = np.sum(sample_weight * (y - np.average(y, weights=sample_weight)) ** 2)
        else:
            numerator = np.sum((y - y_pred) ** 2)
            denominator = np.sum((y - np.mean(y)) ** 2)
        
        if denominator == 0.0:
            # All y values are identical
            return 1.0 if numerator == 0.0 else 0.0
        
        return 1.0 - (numerator / denominator)


class ClassifierMixin:
    """
    Mixin class for all classification estimators.
    
    PROVIDES:
    - score() -> accuracy
    - predict_proba() interface
    - predict_log_proba() interface
    """
    
    _estimator_type = "classifier"
    
    def score(self, X: np.ndarray, y: np.ndarray,
              sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Accuracy score: fraction of correct predictions.
        
        FORMULA: accuracy = (# correct) / (# total)
        
        THE GOTCHA: Accuracy is misleading for imbalanced datasets!
        
        Example: 95% of samples are class 0
        A model that ALWAYS predicts 0 gets 95% accuracy!
        
        Better metrics for imbalanced data:
        - F1 score
        - ROC-AUC
        - Precision-Recall curves
        """
        from sklearn.metrics import accuracy_score  # We'll implement this later
        y_pred = self.predict(X)
        return np.average(y_pred == y, weights=sample_weight)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        NOT ALL CLASSIFIERS SUPPORT THIS!
        - Logistic Regression: Yes (sigmoid outputs)
        - SVM: No (decision function, not probabilities)
        - Decision Tree: Yes (class proportions in leaf)
        - k-NN: Yes (fraction of neighbors)
        
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        
        THE CALIBRATION GOTCHA:
        Just because a model outputs probabilities doesn't mean they're
        well-calibrated! Tree models especially need calibration.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_proba"
        )
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Log of probability estimates.
        
        WHY THIS EXISTS:
        Numerical stability! log(very_small_number) is more stable than very_small_number.
        
        Example:
            proba = 1e-100
            log_proba = -230.26  # Much more stable to work with
        
        Used in:
        - Naive Bayes (log probabilities prevent underflow)
        - Calibration methods
        """
        return np.log(self.predict_proba(X))


class TransformerMixin:
    """
    Mixin for all transformers (scalers, encoders, etc.)
    
    PROVIDES:
    - fit_transform() convenience method
    
    THE PATTERN:
        class StandardScaler(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                # Learn mean and std
                return self
            
            def transform(self, X):
                # Apply scaling
                return X_scaled
        
        # Now you get fit_transform() for free!
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # One line instead of two
    """
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                      **fit_params) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        THE OPTIMIZATION:
        Some estimators can do fit+transform more efficiently than
        fit() then transform() separately.
        
        Example: PCA
        - fit() computes eigenvectors
        - transform() projects data
        - fit_transform() can do both in one SVD decomposition!
        """
        return self.fit(X, y, **fit_params).transform(X)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class NotFittedError(ValueError, AttributeError):
    """
    Exception raised when calling predict on unfitted estimator.
    
    WHY INHERIT FROM BOTH ValueError AND AttributeError?
    
    Historical reasons in sklearn - some old code expects AttributeError
    (from hasattr checks), newer code expects ValueError (from validation).
    
    Multiple inheritance keeps both happy.
    """
    pass


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Let's create a minimal working estimator
    
    class DummyRegressor(BaseEstimator, RegressorMixin):
        """Always predicts the mean of training data."""
        
        def __init__(self, strategy: str = 'mean'):
            self.strategy = strategy
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            X, y = self._validate_data(X, y, reset=True)
            
            if self.strategy == 'mean':
                self.constant_ = np.mean(y)
            elif self.strategy == 'median':
                self.constant_ = np.median(y)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            self._is_fitted = True
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            self._check_is_fitted()
            X = self._validate_data(X, reset=False)
            return np.full(len(X), self.constant_)
    
    # Test it
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    model = DummyRegressor(strategy='mean')
    print(f"Before fit: {model}")
    print(f"Params: {model.get_params()}")
    
    try:
        model.predict(X)
    except NotFittedError as e:
        print(f"\nCaught expected error: {e}")
    
    # Fit and predict
    model.fit(X, y)
    predictions = model.predict(X)
    # score = model.score(X, y)
    
    print(f"\nAfter fit: {model}")
    print(f"Learned constant: {model.constant_:.3f}")
    # print(f"R² score: {score:.3f}")
    
    # Test method chaining
    predictions = DummyRegressor().fit(X, y).predict(X)
    print(f"\nMethod chaining works: {len(predictions)} predictions")
    
    # Test cloning
    clone = model.clone()
    print(f"\nCloned model (unfitted): {clone}")
    print(f"Has constant_? {hasattr(clone, 'constant_')}")