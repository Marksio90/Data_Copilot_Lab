"""
Data Copilot Lab - AutoML
Automated Machine Learning pipeline
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.modules.ml.classification import ClassificationTrainer, ClassifierType
from src.modules.ml.regression import RegressionTrainer, RegressorType
from src.modules.ml.feature_engineering import FeatureEngineer
from src.modules.data_cleaning.missing_handler import MissingDataHandler
from src.modules.data_cleaning.standardizer import DataStandardizer, ScalingMethod
from src.core.exceptions import InvalidParameterError, ModelTrainingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TaskType(str, Enum):
    """ML task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"  # Automatically detect task type


class AutoML:
    """
    Automated Machine Learning Pipeline

    Automatically handles:
    - Task type detection (classification vs regression)
    - Data preprocessing and cleaning
    - Feature engineering and selection
    - Model selection and training
    - Hyperparameter tuning
    - Model evaluation and comparison
    """

    def __init__(self, random_state: int = 42, time_budget: Optional[int] = None):
        """
        Initialize AutoML

        Args:
            random_state: Random seed
            time_budget: Maximum time in seconds (None = no limit)
        """
        self.logger = logger
        self.random_state = random_state
        self.time_budget = time_budget

        self.task_type = None
        self.target_column = None
        self.feature_columns = None
        self.best_model = None
        self.best_model_type = None
        self.best_score = None
        self.pipeline_steps = []
        self.all_results = {}

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        task_type: Union[str, TaskType] = TaskType.AUTO,
        metric: Optional[str] = None,
        models_to_try: Optional[List[str]] = None,
        feature_engineering: bool = True,
        hyperparameter_tuning: bool = True,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Fit AutoML pipeline

        Args:
            data: DataFrame with features and target
            target: Target column name
            task_type: 'classification', 'regression', or 'auto'
            metric: Optimization metric (None = default for task)
            models_to_try: List of model types to try (None = all)
            feature_engineering: Apply feature engineering
            hyperparameter_tuning: Tune hyperparameters
            test_size: Test set proportion
            cv_folds: Cross-validation folds

        Returns:
            AutoML results including best model and performance
        """
        self.logger.info("🚀 Starting AutoML pipeline...")

        # Validate input
        if target not in data.columns:
            raise InvalidParameterError(f"Target column '{target}' not found")

        self.target_column = target

        # Convert task_type to enum
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        # Detect task type if auto
        if task_type == TaskType.AUTO:
            task_type = self._detect_task_type(data, target)
            self.logger.info(f"✓ Detected task type: {task_type.value}")

        self.task_type = task_type

        # Set default metric
        if metric is None:
            metric = 'accuracy' if task_type == TaskType.CLASSIFICATION else 'r2'

        # Step 1: Data Preprocessing
        self.logger.info("📊 Step 1: Data Preprocessing")
        data_clean = self._preprocess_data(data, target)

        # Step 2: Feature Engineering
        if feature_engineering:
            self.logger.info("🔧 Step 2: Feature Engineering")
            data_engineered = self._apply_feature_engineering(data_clean, target)
        else:
            data_engineered = data_clean

        # Get feature columns
        self.feature_columns = [col for col in data_engineered.columns if col != target]

        # Step 3: Model Selection and Training
        self.logger.info(f"🤖 Step 3: Training multiple {task_type.value} models")

        if models_to_try is None:
            models_to_try = self._get_default_models(task_type)

        # Train and evaluate all models
        for model_type in models_to_try:
            try:
                self.logger.info(f"  Training {model_type}...")

                if task_type == TaskType.CLASSIFICATION:
                    results = self._train_classifier(
                        data_engineered, target, model_type, test_size, cv_folds
                    )
                else:
                    results = self._train_regressor(
                        data_engineered, target, model_type, test_size, cv_folds
                    )

                self.all_results[model_type] = results

                # Track best model
                score = results['test_metrics'].get(metric, 0)
                if self.best_score is None or self._is_better_score(score, self.best_score, metric):
                    self.best_score = score
                    self.best_model_type = model_type
                    self.best_model = results['model']

                self.logger.info(f"  ✓ {model_type}: {metric}={score:.4f}")

            except Exception as e:
                self.logger.warning(f"  ✗ Failed to train {model_type}: {e}")

        # Step 4: Hyperparameter Tuning (for best model)
        if hyperparameter_tuning and self.best_model_type:
            self.logger.info(f"⚙️ Step 4: Tuning hyperparameters for {self.best_model_type}")
            tuned_results = self._tune_best_model(data_engineered, target, cv_folds, metric)

            if tuned_results and tuned_results['best_score'] > self.best_score:
                self.logger.info(f"  ✓ Tuning improved score: {tuned_results['best_score']:.4f}")
                # Retrain with best params
                if task_type == TaskType.CLASSIFICATION:
                    results = self._train_classifier(
                        data_engineered, target, self.best_model_type,
                        test_size, cv_folds, tuned_results['best_params']
                    )
                else:
                    results = self._train_regressor(
                        data_engineered, target, self.best_model_type,
                        test_size, cv_folds, tuned_results['best_params']
                    )

                self.best_model = results['model']
                self.best_score = results['test_metrics'].get(metric, self.best_score)

        # Generate report
        report = self._generate_report(metric)

        self.logger.info(f"✅ AutoML complete! Best model: {self.best_model_type} ({metric}={self.best_score:.4f})")

        return report

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with best model

        Args:
            data: DataFrame with features

        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ModelTrainingError("No model trained yet. Call fit() first.")

        return self.best_model.predict(data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (classification only)

        Args:
            data: DataFrame with features

        Returns:
            Prediction probabilities
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise InvalidParameterError("predict_proba only available for classification")

        if self.best_model is None:
            raise ModelTrainingError("No model trained yet. Call fit() first.")

        return self.best_model.predict_proba(data)

    def _detect_task_type(self, data: pd.DataFrame, target: str) -> TaskType:
        """Automatically detect if task is classification or regression"""
        target_series = data[target]

        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(target_series):
            return TaskType.CLASSIFICATION

        # Check number of unique values
        n_unique = target_series.nunique()
        n_samples = len(target_series)

        # Heuristic: if unique values < 20 and < 5% of samples, likely classification
        if n_unique < 20 and n_unique < 0.05 * n_samples:
            return TaskType.CLASSIFICATION

        # Check if values are integers (another hint for classification)
        if target_series.dtype in ['int64', 'int32'] and n_unique < 50:
            return TaskType.CLASSIFICATION

        return TaskType.REGRESSION

    def _preprocess_data(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Preprocess data: handle missing values and standardize"""
        result = data.copy()

        # Handle missing values
        missing_handler = MissingDataHandler()
        missing_info = missing_handler.analyze(result)

        if missing_info['total_missing'] > 0:
            self.logger.info(f"  Handling {missing_info['total_missing']} missing values...")

            # Use appropriate strategy per column
            for col in result.columns:
                if result[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(result[col]):
                        result = missing_handler.clean(result, strategy='fill_median', columns=[col])
                    else:
                        result = missing_handler.clean(result, strategy='fill_mode', columns=[col])

        # Standardize numeric features (except target)
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        if numeric_cols:
            standardizer = DataStandardizer()
            result = standardizer.scale_numeric(result, columns=numeric_cols, method=ScalingMethod.STANDARD)

        self.pipeline_steps.append({
            "step": "preprocessing",
            "missing_handled": missing_info['total_missing'],
            "features_standardized": len(numeric_cols)
        })

        return result

    def _apply_feature_engineering(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Apply automatic feature engineering"""
        engineer = FeatureEngineer()
        result = data.copy()

        # Extract date features if any datetime columns
        datetime_cols = result.select_dtypes(include=['datetime']).columns.tolist()
        for col in datetime_cols:
            if col != target:
                result = engineer.extract_date_features(result, col)

        # Remove original datetime columns
        result = result.drop(columns=datetime_cols)

        # Low variance feature removal
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        if len(numeric_cols) > 10:  # Only if we have many features
            original_n = len(numeric_cols)
            result = engineer.select_by_variance(result, threshold=0.01, columns=numeric_cols)
            removed = original_n - (len(result.columns) - 1)  # -1 for target
            if removed > 0:
                self.logger.info(f"  Removed {removed} low-variance features")

        self.pipeline_steps.append({
            "step": "feature_engineering",
            "datetime_features_extracted": len(datetime_cols),
            "low_variance_removed": removed if 'removed' in locals() else 0
        })

        return result

    def _get_default_models(self, task_type: TaskType) -> List[str]:
        """Get default list of models to try"""
        if task_type == TaskType.CLASSIFICATION:
            return [
                'logistic_regression',
                'random_forest',
                'gradient_boosting',
                'xgboost',
                'svm'
            ]
        else:  # REGRESSION
            return [
                'linear_regression',
                'ridge',
                'random_forest',
                'gradient_boosting',
                'xgboost'
            ]

    def _train_classifier(
        self,
        data: pd.DataFrame,
        target: str,
        model_type: str,
        test_size: float,
        cv_folds: int,
        hyperparameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Train a classification model"""
        trainer = ClassificationTrainer(random_state=self.random_state)

        results = trainer.train(
            data=data,
            target=target,
            model_type=model_type,
            test_size=test_size,
            hyperparameters=hyperparameters,
            cross_validate=True,
            cv_folds=cv_folds
        )

        results['model'] = trainer

        return results

    def _train_regressor(
        self,
        data: pd.DataFrame,
        target: str,
        model_type: str,
        test_size: float,
        cv_folds: int,
        hyperparameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Train a regression model"""
        trainer = RegressionTrainer(random_state=self.random_state)

        results = trainer.train(
            data=data,
            target=target,
            model_type=model_type,
            test_size=test_size,
            hyperparameters=hyperparameters,
            cross_validate=True,
            cv_folds=cv_folds
        )

        results['model'] = trainer

        return results

    def _tune_best_model(
        self,
        data: pd.DataFrame,
        target: str,
        cv_folds: int,
        metric: str
    ) -> Optional[Dict[str, Any]]:
        """Tune hyperparameters for best model"""
        if not self.best_model_type:
            return None

        try:
            if self.task_type == TaskType.CLASSIFICATION:
                trainer = ClassificationTrainer(random_state=self.random_state)
                results = trainer.tune_hyperparameters(
                    data=data,
                    target=target,
                    model_type=self.best_model_type,
                    search_type='random',
                    cv_folds=cv_folds,
                    n_iter=10,
                    scoring=metric
                )
            else:
                trainer = RegressionTrainer(random_state=self.random_state)
                results = trainer.tune_hyperparameters(
                    data=data,
                    target=target,
                    model_type=self.best_model_type,
                    search_type='random',
                    cv_folds=cv_folds,
                    n_iter=10,
                    scoring=metric
                )

            return results

        except Exception as e:
            self.logger.warning(f"Hyperparameter tuning failed: {e}")
            return None

    def _is_better_score(self, new_score: float, current_score: float, metric: str) -> bool:
        """Check if new score is better than current"""
        # For most metrics, higher is better
        higher_is_better = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'r2']

        # For some metrics, lower is better
        lower_is_better = ['mse', 'rmse', 'mae', 'log_loss']

        if metric in higher_is_better:
            return new_score > current_score
        elif metric in lower_is_better:
            return new_score < current_score
        else:
            # Default: higher is better
            return new_score > current_score

    def _generate_report(self, metric: str) -> Dict[str, Any]:
        """Generate comprehensive AutoML report"""
        # Rank all models by performance
        model_rankings = []
        for model_type, results in self.all_results.items():
            score = results['test_metrics'].get(metric, 0)
            model_rankings.append({
                "model": model_type,
                "score": score,
                "metrics": results['test_metrics']
            })

        # Sort by score
        model_rankings.sort(key=lambda x: x['score'], reverse=True)

        report = {
            "task_type": self.task_type.value,
            "target": self.target_column,
            "n_features": len(self.feature_columns),
            "optimization_metric": metric,
            "best_model": {
                "type": self.best_model_type,
                "score": self.best_score,
                "full_results": self.all_results.get(self.best_model_type, {})
            },
            "model_rankings": model_rankings,
            "pipeline_steps": self.pipeline_steps,
            "all_models_tried": len(self.all_results)
        }

        return report

    def get_best_model(self):
        """Get the best trained model"""
        return self.best_model

    def get_all_results(self) -> Dict[str, Any]:
        """Get results from all trained models"""
        return self.all_results

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from best model"""
        if self.best_model and hasattr(self.best_model, 'get_training_history'):
            history = self.best_model.get_training_history()
            return history.get('feature_importance')
        return None
