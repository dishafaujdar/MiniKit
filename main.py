"""
MiniKit: Clean-room ML implementations for foundational understanding
Architecture inspired by scikit-learn's design patterns
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union

# BASE CLASSES (The Foundation)

class BaseEstimator(ABC):
    """Base class for all estimators. Provides the contract."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Learn from data. MUST return self for method chaining."""
        pass

    # In your file: BaseEstimator.predict is abstract. That ensures any estimator (LinearRegression, LogisticRegression, DecisionTreeClassifier) must provide a predict implementation — otherwise attempting to instantiate the class raises TypeError.

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions. MUST check if fitted first."""
        pass
    
    def _check_is_fitted(self):
        """sklearn's pattern - prevents predict before fit."""
        if not hasattr(self, '_is_fitted'):
            raise RuntimeError("Call fit() before predict()!")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Default scoring - override for classification vs regression."""
        raise NotImplementedError

# MIXINS: Add Behavior Without Deep Inheritance
"RegressorMixin → adds score() that calculates R² (for continuous predictions)"
class RegressorMixin:
    """Mixin for regression-specific behavior."""
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score for regressors."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

"ClassifierMixin → adds score() that calculates accuracy (for discrete predictions)"
class ClassifierMixin:
    """Mixin for classification-specific behavior."""
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy for classifiers."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates - not all classifiers support this."""
        raise NotImplementedError(f"{self.__class__.__name__} doesn't support predict_proba")


# LINEAR MODELS

class LinearRegression(RegressorMixin,BaseEstimator):
    
    def __init__(self,l_r: float = 0.1, epoch: int = 1000):
        self.l_r = l_r
        self.epochs = epoch
        self.weights = None
        self.bias = None

    def fit(self,X: np.ndarray, y:np.ndarray):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features) #as in the starting we consider all weights 0
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X,self.weights) + self.bias

            dW = (1 / n_samples) * np.dot(X.T, (y_pred - y)) #derivate of cost fn MSE w.r.t weight (THE SLOPE)
            dB = (1 / n_samples) * np.sum(y_pred - y) #derivate of cost fn MSE w.r.t bias (THE SLOPE)

            self.weights = self.weights - self.l_r * dW
            self.bias = self.bias - self.l_r * dB

        self._is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return np.dot(X,self.weights) + self.bias
    

class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, l_r: float = 0.01, epochs: int = 1000, fit_intercept: bool = True):
        self.l_r = l_r
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []  # Track convergence

    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))


    def fit(self,X: np.ndarray,y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            linear_output = np.dot(X,self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.l_r * dw
            self.bias = self.bias - self.l_r * db

            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
            self.loss_history_.append(loss)

        return self


    # predict_proba gives you probabilities (between 0 and 1)
    def predict_proba(self, X: np.ndarray):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X: np.ndarray):
        y_pred_proba = self.predict_proba(X)
        return np.where(y_pred_proba >= 0.5, 1, 0)
    

# TREE MODELS

class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    CART algorithm for classification.
    
    THE KEY INSIGHT: Trees recursively partition feature space by
    maximizing information gain (minimizing impurity).
    
    This is where you learn WHY trees are so powerful:
    - Non-linear decision boundaries for free
    - No feature scaling needed
    - Interpretable rules
    
    GOTCHA: Easy to overfit! That's why everyone uses ensembles.
    """
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Gini = 1 - Σ(p_i²)
        
        INTUITION: Probability of misclassifying a random sample
        if we randomly assign labels based on class distribution.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _information_gain(self, X_column: np.ndarray, y: np.ndarray, 
                         threshold: float) -> float:
        """IG = Parent_Impurity - Weighted_Avg(Children_Impurity)"""
        parent_impurity = self._gini_impurity(y)
        
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        n = len(y)
        left_impurity = self._gini_impurity(y[left_mask])
        right_impurity = self._gini_impurity(y[right_mask])
        
        child_impurity = (np.sum(left_mask) / n * left_impurity + 
                         np.sum(right_mask) / n * right_impurity)
        
        return parent_impurity - child_impurity
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Brute-force search over all features and thresholds.
        
        OPTIMIZATION OPPORTUNITY: This is O(n_features * n_samples * log(n_samples))
        Real implementations use histograms (XGBoost) or approximations (LightGBM).
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(X[:, feature_idx], y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Recursive tree building."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or n_samples < self.min_samples_split:
            # Leaf node: return most common class
            leaf_value = np.argmax(np.bincount(y))
            return {'leaf': True, 'value': leaf_value}
        
        # Find best split
        feature_idx, threshold = self._find_best_split(X, y)
        
        if feature_idx is None:
            leaf_value = np.argmax(np.bincount(y))
            return {'leaf': True, 'value': leaf_value}
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree_ = self._build_tree(X, y)
        self._is_fitted = True
        return self
    
    def _predict_sample(self, x: np.ndarray, tree: dict) -> int:
        """Traverse tree for a single sample."""
        if tree['leaf']:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return np.array([self._predict_sample(x, self.tree_) for x in X])


if __name__ == "__main__":
    # Generate toy data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Train models
    lr = LinearRegression(l_r=1.0, epoch=1000).fit(X, y)
    lr_preds = lr.predict(X)
    lr_accuracy = np.mean((lr_preds > 0.5).astype(int) == y)
    print(f"Linear Regression Accuracy (thresholded): {lr_accuracy:.3f}")
    print(f"→ Learned weights: {lr.weights}")
    print(f"→ True weights: [1.0, 1.0] (from y = x₁ + x₂ > 0)")
    print(f"→ Prediction range: [{lr_preds.min():.2f}, {lr_preds.max():.2f}]")
    
    logreg = LogisticRegression(l_r=0.1, epochs=1000).fit(X, y)
    log_preds = logreg.predict(X)
    log_acc = np.mean(log_preds == y)
    print(f"Logistic Regression Accuracy: {log_acc:.3f}")
    print(f"→ Final loss: {logreg.loss_history_[-1]:.6f}")

    # tree = DecisionTreeClassifier(max_depth=3)
    # tree.fit(X, y)
    # print(f"Decision Tree Accuracy: {tree.score(X, y):.3f}")