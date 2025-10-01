"""
MiniKit: Clean-room ML implementations for foundational understanding
Architecture inspired by scikit-learn's design patterns
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union
from graphviz import Digraph
 
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
    
    def _softmax(self,s):
        return 

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
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    
    def _gini_impurity(self, y: np.ndarray) -> float:
        "the probability of misclassification"
        m = len(y)
        if m == 0:
            return 0
        
        prob = np.bincount(y) / m #counts how many samples belong to each class
        return 1 - np.sum(prob ** 2) #foramula to find the gini impurity
    

    def _information_gain(self, X_column: np.ndarray, y: np.ndarray, threshold: float) -> float:
        "the measure of DISORDER OR compute the info gain from potential split."
        parent_impurity = self._gini_impurity(y)
        m = len(y)

        # split
        left_mark = X_column <= threshold
        right_mark = ~left_mark

        if np.sum(left_mark) == 0 or np.sum(right_mark) == 0:
            return 0 #no valid split
        
        # weighted avg impurity of childern
        n_left,n_right = np.sum(left_mark), np.sum(right_mark)
        left_impurity = self._gini_impurity(y[left_mark])
        right_impurity = self._gini_impurity(y[right_mark])
        child_impurity = (n_left/m) * left_impurity + (n_right/m) * right_impurity

        # info gain
        return parent_impurity - child_impurity
    

    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        "the best feature and threshold to split on"
        best_gain = -1
        split_idx, split_threshold = None, None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            X_column = X[:,feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(X_column,y,threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        print("feature_idx:", feature_idx, type(feature_idx))

        return split_idx, split_threshold, best_gain


    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        "using recursion"
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # when to stop
        if(self.max_depth  is not None and depth >= self.max_depth) or \
        num_labels == 1 or \
        num_samples < self.min_samples_split:
            # leaf node return majority class
            leaf_value = np.bincount(y).argmax()
            return {"leaf":True,"class": leaf_value}
        
        # find the best split
        feature_idx,threshold,gain = self._find_best_split(X,y)

        if feature_idx is None or gain == 0:
            leaf_value = np.bincount(y).argmax()
            return {"leaf":True,"class": leaf_value}
        
        # split data
        left_mask = X[:,feature_idx] <= threshold
        rigt_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask],y[left_mask],depth+1)
        right_subtree = self._build_tree(X[rigt_mask],y[rigt_mask],depth+1)

        return {
            "leaf": False,
            "feature": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }


    def fit(self, X: np.ndarray, y: np.ndarray):
        print(f"Training on {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {np.unique(y)}")
        self.tree_ = self._build_tree(X, y)
        print(f"Tree built: {self.tree_ is not None}")
        print(f"Tree root: {self.tree_}")  # See what you actually built
        return self 
  

    def _predict_sample(self, x: np.ndarray, tree: dict) -> int:
        "tree traversing"
        if tree is None:
            raise ValueError ("Tree is None - model was not fitted properly")
        
        if tree["leaf"]:
            return tree["class"]
        
        feature_val = x[tree["feature"]]
        if feature_val <= tree["threshold"]:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])

    
    def predict(self, X: np.ndarray) -> np.ndarray:
            if self.tree_ is None:
                raise ValueError("This DecisionTreeClassifier instance is not fitted yet. "
                        "Call 'fit' with appropriate arguments before using predict.")

            return np.array([self._predict_sample(sample,self.tree_) for sample in X])


    def plot_tree(self,tree, feature_names=None, node=None, graph=None, parent=None, edge_label=""):
        if graph is None:
            graph = Digraph()
            graph.attr(rankdir='TB')  # Top to Bottom layout
            node = 0
        
        # Leaf node
        if tree["leaf"]:
            label = f"Leaf\nClass={tree['class']}"
            graph.node(str(node), label=label, shape="box", style="filled", color="lightblue")
            if parent is not None:
                graph.edge(str(parent), str(node), label=edge_label)
            return graph, node + 1

        # Decision node
        feat_name = feature_names[tree["feature"]] if feature_names else f"X{tree['feature']}"
        label = f"{feat_name} <= {tree['threshold']:.2f}"
        graph.node(str(node), label=label, shape="ellipse",style="filled", fillcolor="lightgray")
        
        if parent is not None:
            graph.edge(str(parent), str(node), label=edge_label)

        curr_node = node
        next_node = curr_node + 1

        # Left subtree
        graph, next_node = self.plot_tree(tree["left"], feature_names, next_node, 
                                        graph, curr_node, "True")
        graph, next_node = self.plot_tree(tree["right"], feature_names, next_node, 
                                        graph, curr_node, "False")

        return graph, next_node


class Knn():
    pass

class EnsembleLearning():
    pass

class TheRLERegressor():
    pass


if __name__ == "__main__":

    # Generate toy data
    # np.random.seed(42)
    # X = np.random.randn(100, 2)
    # y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # # Train models
    # lr = LinearRegression(l_r=1.0, epoch=1000).fit(X, y)
    # lr_preds = lr.predict(X)
    # lr_accuracy = np.mean((lr_preds > 0.5).astype(int) == y)
    # print(f"Linear Regression Accuracy (thresholded): {lr_accuracy:.3f}")
    # print(f"→ Learned weights: {lr.weights}")
    # print(f"→ True weights: [1.0, 1.0] (from y = x₁ + x₂ > 0)")
    # print(f"→ Prediction range: [{lr_preds.min():.2f}, {lr_preds.max():.2f}]")
    
    # logreg = LogisticRegression(l_r=0.1, epochs=1000).fit(X, y)
    # log_preds = logreg.predict(X)
    # log_acc = np.mean(log_preds == y)
    # print(f"Logistic Regression Accuracy: {log_acc:.3f}")
    # print(f"→ Final loss: {logreg.loss_history_[-1]:.6f}")

    # Simple train/test split
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train our toy CART classifier
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)

    # Predict
    y_pred = tree.predict(X_test)
    print("Classification accuracy:", accuracy_score(y_test, y_pred))
    feature_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    graph, _ = tree.plot_tree(tree.tree_, feature_names)
    graph.render("decision_tree", format="png", cleanup=True)  # Saves as decision_tree.png
    graph.view()  # Opens the image
