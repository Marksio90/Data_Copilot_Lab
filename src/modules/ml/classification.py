"""
Data Copilot Lab - Classification Models
Comprehensive classification model training and prediction
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss,
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.core.exceptions import InvalidParameterError, ModelTrainingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClassifierType(str, Enum):
    """Available classifier types"""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"
    LINEAR_SVM = "linear_svm"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    DECISION_TREE = "decision_tree"
    ADABOOST = "adaboost"
    EXTRA_TREES = "extra_trees"
    MLP = "mlp"


class ClassificationTrainer:
    """
    Train and evaluate classification models

    Supports 10+ algorithms with hyperparameter tuning,
    cross-validation, and comprehensive evaluation
    """

    def __init__(self, random_state: int = 42):
        self.logger = logger
        self.random_state = random_state
        self.model = None
        self.model_type = None
        self.feature_names = None
        self.target_name = None
        self.classes_ = None
        self.training_history = {}

    def train(
        self,
        data: pd.DataFrame,
        target: str,
        model_type: Union[str, ClassifierType] = ClassifierType.RANDOM_FOREST,
        test_size: float = 0.2,
        features: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        cross_validate: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train a classification model

        Args:
            data: DataFrame with features and target
            target: Target column name
            model_type: Type of classifier to train
            test_size: Test set proportion
            features: Feature columns (None = all except target)
            hyperparameters: Model hyperparameters
            cross_validate: Perform cross-validation
            cv_folds: Number of CV folds

        Returns:
            Training results and metrics
        """
        if target not in data.columns:
            raise InvalidParameterError(f"Target '{target}' not found")

        # Convert to enum
        if isinstance(model_type, str):
            model_type = ClassifierType(model_type)

        self.model_type = model_type
        self.target_name = target

        self.logger.info(f"Training {model_type.value} classifier")

        # Prepare data
        if features is None:
            features = [col for col in data.columns if col != target]

        self.feature_names = features

        X = data[features]
        y = data[target]

        # Handle categorical features (simple label encoding for now)
        X = self._prepare_features(X)

        # Store classes
        self.classes_ = np.unique(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Create model
        self.model = self._create_model(model_type, hyperparameters)

        # Train
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            raise ModelTrainingError(f"Training failed: {str(e)}")

        # Evaluate
        train_metrics = self._evaluate(X_train, y_train, "train")
        test_metrics = self._evaluate(X_test, y_test, "test")

        # Cross-validation
        cv_scores = None
        if cross_validate:
            cv_scores = cross_val_score(
                self.model, X_train, y_train, cv=cv_folds, scoring='accuracy'
            )
            self.logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Feature importance
        feature_importance = self._get_feature_importance()

        # Store training history
        self.training_history = {
            "model_type": model_type.value,
            "n_features": len(features),
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
            "feature_importance": feature_importance,
            "hyperparameters": hyperparameters or {}
        }

        self.logger.info(f"Training complete. Test accuracy: {test_metrics['accuracy']:.4f}")

        return self.training_history

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            data: DataFrame with features

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ModelTrainingError("Model not trained yet")

        X = data[self.feature_names]
        X = self._prepare_features(X)

        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            data: DataFrame with features

        Returns:
            Predicted probabilities for each class
        """
        if self.model is None:
            raise ModelTrainingError("Model not trained yet")

        if not hasattr(self.model, 'predict_proba'):
            raise InvalidParameterError(f"{self.model_type.value} does not support probability prediction")

        X = data[self.feature_names]
        X = self._prepare_features(X)

        probabilities = self.model.predict_proba(X)

        return probabilities

    def tune_hyperparameters(
        self,
        data: pd.DataFrame,
        target: str,
        model_type: Union[str, ClassifierType],
        param_grid: Optional[Dict[str, List]] = None,
        search_type: str = 'grid',
        cv_folds: int = 5,
        n_iter: int = 10,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters

        Args:
            data: DataFrame
            target: Target column
            model_type: Model type
            param_grid: Parameter grid (None = default grid)
            search_type: 'grid' or 'random'
            cv_folds: CV folds
            n_iter: Iterations for random search
            scoring: Scoring metric

        Returns:
            Best parameters and scores
        """
        if isinstance(model_type, str):
            model_type = ClassifierType(model_type)

        self.logger.info(f"Tuning hyperparameters for {model_type.value}")

        # Prepare data
        features = [col for col in data.columns if col != target]
        X = data[features]
        y = data[target]
        X = self._prepare_features(X)

        # Get default param grid if not provided
        if param_grid is None:
            param_grid = self._get_default_param_grid(model_type)

        # Create base model
        base_model = self._create_model(model_type, None)

        # Search
        if search_type == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )

        search.fit(X, y)

        results = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "cv_results": {
                "mean_test_score": search.cv_results_['mean_test_score'].tolist(),
                "std_test_score": search.cv_results_['std_test_score'].tolist(),
                "params": [str(p) for p in search.cv_results_['params']]
            }
        }

        self.logger.info(f"Best score: {search.best_score_:.4f}")
        self.logger.info(f"Best params: {search.best_params_}")

        return results

    def _create_model(
        self,
        model_type: ClassifierType,
        hyperparameters: Optional[Dict[str, Any]]
    ):
        """Create model instance"""
        params = hyperparameters or {}

        if model_type == ClassifierType.LOGISTIC_REGRESSION:
            return LogisticRegression(random_state=self.random_state, max_iter=1000, **params)

        elif model_type == ClassifierType.RANDOM_FOREST:
            return RandomForestClassifier(random_state=self.random_state, **params)

        elif model_type == ClassifierType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(random_state=self.random_state, **params)

        elif model_type == ClassifierType.XGBOOST:
            if not XGBOOST_AVAILABLE:
                raise InvalidParameterError("XGBoost not installed")
            return xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, **params)

        elif model_type == ClassifierType.LIGHTGBM:
            if not LIGHTGBM_AVAILABLE:
                raise InvalidParameterError("LightGBM not installed")
            return lgb.LGBMClassifier(random_state=self.random_state, verbose=-1, **params)

        elif model_type == ClassifierType.SVM:
            return SVC(random_state=self.random_state, probability=True, **params)

        elif model_type == ClassifierType.LINEAR_SVM:
            return LinearSVC(random_state=self.random_state, max_iter=2000, **params)

        elif model_type == ClassifierType.KNN:
            return KNeighborsClassifier(**params)

        elif model_type == ClassifierType.NAIVE_BAYES:
            return GaussianNB(**params)

        elif model_type == ClassifierType.DECISION_TREE:
            return DecisionTreeClassifier(random_state=self.random_state, **params)

        elif model_type == ClassifierType.ADABOOST:
            return AdaBoostClassifier(random_state=self.random_state, **params)

        elif model_type == ClassifierType.EXTRA_TREES:
            return ExtraTreesClassifier(random_state=self.random_state, **params)

        elif model_type == ClassifierType.MLP:
            return MLPClassifier(random_state=self.random_state, max_iter=500, **params)

        else:
            raise InvalidParameterError(f"Unknown model type: {model_type}")

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training"""
        X = X.copy()

        # Handle categorical features with simple label encoding
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = pd.Categorical(X[col]).codes

        # Fill missing values
        X = X.fillna(X.mean(numeric_only=True))

        return X

    def _evaluate(self, X: pd.DataFrame, y: pd.Series, split: str) -> Dict[str, Any]:
        """Evaluate model on given data"""
        y_pred = self.model.predict(X)

        # Basic metrics
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y, y_pred, average='weighted', zero_division=0))
        }

        # ROC AUC (for binary or multiclass with probabilities)
        if hasattr(self.model, 'predict_proba'):
            try:
                y_proba = self.model.predict_proba(X)
                if len(self.classes_) == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y, y_proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y, y_proba, multi_class='ovr'))
                metrics["log_loss"] = float(log_loss(y, y_proba))
            except Exception as e:
                self.logger.debug(f"Could not calculate ROC AUC: {e}")

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report

        return metrics

    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if not hasattr(self.model, 'feature_importances_'):
            return None

        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def _get_default_param_grid(self, model_type: ClassifierType) -> Dict[str, List]:
        """Get default hyperparameter grid for tuning"""
        grids = {
            ClassifierType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            ClassifierType.LOGISTIC_REGRESSION: {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            ClassifierType.GRADIENT_BOOSTING: {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            ClassifierType.SVM: {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            ClassifierType.KNN: {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }

        return grids.get(model_type, {})

    def get_model(self):
        """Get trained model"""
        return self.model

    def get_training_history(self) -> Dict[str, Any]:
        """Get training history and metrics"""
        return self.training_history
