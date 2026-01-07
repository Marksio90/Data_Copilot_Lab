"""
Data Copilot Lab - Regression Models
Comprehensive regression model training and prediction
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    BayesianRidge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
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


class RegressorType(str, Enum):
    """Available regressor types"""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVR = "svr"
    LINEAR_SVR = "linear_svr"
    KNN = "knn"
    DECISION_TREE = "decision_tree"
    ADABOOST = "adaboost"
    EXTRA_TREES = "extra_trees"
    MLP = "mlp"
    BAYESIAN_RIDGE = "bayesian_ridge"


class RegressionTrainer:
    """
    Train and evaluate regression models

    Supports 15+ algorithms with hyperparameter tuning,
    cross-validation, and comprehensive evaluation
    """

    def __init__(self, random_state: int = 42):
        self.logger = logger
        self.random_state = random_state
        self.model = None
        self.model_type = None
        self.feature_names = None
        self.target_name = None
        self.training_history = {}

    def train(
        self,
        data: pd.DataFrame,
        target: str,
        model_type: Union[str, RegressorType] = RegressorType.RANDOM_FOREST,
        test_size: float = 0.2,
        features: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        cross_validate: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train a regression model

        Args:
            data: DataFrame with features and target
            target: Target column name
            model_type: Type of regressor to train
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
            model_type = RegressorType(model_type)

        self.model_type = model_type
        self.target_name = target

        self.logger.info(f"Training {model_type.value} regressor")

        # Prepare data
        if features is None:
            features = [col for col in data.columns if col != target]

        self.feature_names = features

        X = data[features]
        y = data[target]

        # Handle categorical features
        X = self._prepare_features(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
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
                self.model, X_train, y_train, cv=cv_folds, scoring='r2'
            )
            self.logger.info(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Feature importance
        feature_importance = self._get_feature_importance()

        # Store training history
        self.training_history = {
            "model_type": model_type.value,
            "n_features": len(features),
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "target_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max())
            },
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
            "feature_importance": feature_importance,
            "hyperparameters": hyperparameters or {}
        }

        self.logger.info(f"Training complete. Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")

        return self.training_history

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            data: DataFrame with features

        Returns:
            Predicted values
        """
        if self.model is None:
            raise ModelTrainingError("Model not trained yet")

        X = data[self.feature_names]
        X = self._prepare_features(X)

        predictions = self.model.predict(X)

        return predictions

    def tune_hyperparameters(
        self,
        data: pd.DataFrame,
        target: str,
        model_type: Union[str, RegressorType],
        param_grid: Optional[Dict[str, List]] = None,
        search_type: str = 'grid',
        cv_folds: int = 5,
        n_iter: int = 10,
        scoring: str = 'r2'
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
            model_type = RegressorType(model_type)

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
        model_type: RegressorType,
        hyperparameters: Optional[Dict[str, Any]]
    ):
        """Create model instance"""
        params = hyperparameters or {}

        if model_type == RegressorType.LINEAR_REGRESSION:
            return LinearRegression(**params)

        elif model_type == RegressorType.RIDGE:
            return Ridge(random_state=self.random_state, **params)

        elif model_type == RegressorType.LASSO:
            return Lasso(random_state=self.random_state, **params)

        elif model_type == RegressorType.ELASTIC_NET:
            return ElasticNet(random_state=self.random_state, **params)

        elif model_type == RegressorType.RANDOM_FOREST:
            return RandomForestRegressor(random_state=self.random_state, **params)

        elif model_type == RegressorType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(random_state=self.random_state, **params)

        elif model_type == RegressorType.XGBOOST:
            if not XGBOOST_AVAILABLE:
                raise InvalidParameterError("XGBoost not installed")
            return xgb.XGBRegressor(random_state=self.random_state, **params)

        elif model_type == RegressorType.LIGHTGBM:
            if not LIGHTGBM_AVAILABLE:
                raise InvalidParameterError("LightGBM not installed")
            return lgb.LGBMRegressor(random_state=self.random_state, verbose=-1, **params)

        elif model_type == RegressorType.SVR:
            return SVR(**params)

        elif model_type == RegressorType.LINEAR_SVR:
            return LinearSVR(random_state=self.random_state, max_iter=2000, **params)

        elif model_type == RegressorType.KNN:
            return KNeighborsRegressor(**params)

        elif model_type == RegressorType.DECISION_TREE:
            return DecisionTreeRegressor(random_state=self.random_state, **params)

        elif model_type == RegressorType.ADABOOST:
            return AdaBoostRegressor(random_state=self.random_state, **params)

        elif model_type == RegressorType.EXTRA_TREES:
            return ExtraTreesRegressor(random_state=self.random_state, **params)

        elif model_type == RegressorType.MLP:
            return MLPRegressor(random_state=self.random_state, max_iter=500, **params)

        elif model_type == RegressorType.BAYESIAN_RIDGE:
            return BayesianRidge(**params)

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

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        explained_var = explained_variance_score(y, y_pred)

        # MAPE (handle division by zero)
        try:
            mape = mean_absolute_percentage_error(y, y_pred)
        except:
            mape = None

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "explained_variance": float(explained_var),
            "mape": float(mape) if mape is not None else None
        }

        # Residuals analysis
        residuals = y - y_pred
        metrics["residuals"] = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max())
        }

        return metrics

    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return importance

        # For linear models, use coefficients
        elif hasattr(self.model, 'coef_'):
            importance = dict(zip(self.feature_names, np.abs(self.model.coef_)))
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return importance

        return None

    def _get_default_param_grid(self, model_type: RegressorType) -> Dict[str, List]:
        """Get default hyperparameter grid for tuning"""
        grids = {
            RegressorType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            RegressorType.RIDGE: {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            RegressorType.LASSO: {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            RegressorType.ELASTIC_NET: {
                'alpha': [0.001, 0.01, 0.1, 1, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            RegressorType.GRADIENT_BOOSTING: {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            RegressorType.SVR: {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            RegressorType.KNN: {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }

        return grids.get(model_type, {})

    def predict_intervals(
        self,
        data: pd.DataFrame,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals (for models that support it)

        Args:
            data: DataFrame with features
            confidence: Confidence level

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        predictions = self.predict(data)

        # For Bayesian models
        if hasattr(self.model, 'predict') and isinstance(self.model, BayesianRidge):
            X = data[self.feature_names]
            X = self._prepare_features(X)
            y_pred, y_std = self.model.predict(X, return_std=True)

            z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
            lower = y_pred - z_score * y_std
            upper = y_pred + z_score * y_std

            return y_pred, lower, upper

        # For ensemble models, use prediction variance
        elif hasattr(self.model, 'estimators_'):
            X = data[self.feature_names]
            X = self._prepare_features(X)

            # Get predictions from all estimators
            all_predictions = np.array([
                estimator.predict(X) for estimator in self.model.estimators_
            ])

            mean_pred = all_predictions.mean(axis=0)
            std_pred = all_predictions.std(axis=0)

            z_score = 1.96 if confidence == 0.95 else 2.576
            lower = mean_pred - z_score * std_pred
            upper = mean_pred + z_score * std_pred

            return mean_pred, lower, upper

        else:
            # No uncertainty estimation available
            self.logger.warning(f"{self.model_type.value} does not support prediction intervals")
            return predictions, predictions, predictions

    def get_model(self):
        """Get trained model"""
        return self.model

    def get_training_history(self) -> Dict[str, Any]:
        """Get training history and metrics"""
        return self.training_history
