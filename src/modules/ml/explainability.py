"""
Data Copilot Lab - Model Explainability
Tools for explaining and interpreting ML model predictions
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import confusion_matrix, classification_report

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """
    Explain and interpret ML model predictions

    Provides multiple explainability methods:
    - Feature importance
    - SHAP values
    - LIME explanations
    - Partial dependence plots
    - Permutation importance
    - Individual prediction explanations
    """

    def __init__(self):
        self.logger = logger
        self.explainers = {}

    def explain_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: Optional[pd.Series] = None,
        method: str = 'all'
    ) -> Dict[str, Any]:
        """
        Comprehensive model explanation

        Args:
            model: Trained sklearn model
            X_train: Training features
            X_test: Test features
            y_test: Test target (optional)
            method: 'feature_importance', 'shap', 'lime', 'permutation', or 'all'

        Returns:
            Dictionary with explanation results
        """
        self.logger.info(f"Explaining model with method: {method}")

        results = {}

        # Feature importance (if available)
        if method in ['feature_importance', 'all']:
            fi = self.get_feature_importance(model, X_train.columns)
            if fi:
                results['feature_importance'] = fi

        # Permutation importance
        if method in ['permutation', 'all'] and y_test is not None:
            perm_importance = self.permutation_feature_importance(model, X_test, y_test)
            results['permutation_importance'] = perm_importance

        # SHAP values
        if method in ['shap', 'all'] and SHAP_AVAILABLE:
            try:
                shap_values = self.calculate_shap_values(model, X_train, X_test)
                results['shap'] = shap_values
            except Exception as e:
                self.logger.warning(f"SHAP calculation failed: {e}")

        # LIME (sample explanation)
        if method in ['lime', 'all'] and LIME_AVAILABLE:
            try:
                # Explain first test instance as example
                lime_exp = self.explain_instance_lime(
                    model, X_train, X_test.iloc[0], X_train.columns
                )
                results['lime_sample'] = lime_exp
            except Exception as e:
                self.logger.warning(f"LIME explanation failed: {e}")

        return results

    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Get feature importance from tree-based models

        Args:
            model: Trained model
            feature_names: Feature names

        Returns:
            Dictionary of {feature: importance}
        """
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return importance

        # For linear models, use coefficients
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim == 1:
                importance = dict(zip(feature_names, np.abs(coef)))
            else:
                # Multi-class: average absolute coefficients
                importance = dict(zip(feature_names, np.abs(coef).mean(axis=0)))

            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return importance

        return None

    def permutation_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate permutation feature importance

        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            n_repeats: Number of permutation repeats

        Returns:
            Dictionary with importance scores
        """
        self.logger.info("Calculating permutation importance...")

        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=42
        )

        importance = {
            "importances_mean": dict(zip(X.columns, result.importances_mean)),
            "importances_std": dict(zip(X.columns, result.importances_std))
        }

        # Sort by mean importance
        importance["importances_mean"] = dict(
            sorted(importance["importances_mean"].items(), key=lambda x: x[1], reverse=True)
        )

        return importance

    def calculate_shap_values(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Calculate SHAP values for model explanations

        Args:
            model: Trained model
            X_train: Training data
            X_test: Test data to explain
            sample_size: Sample size for background data

        Returns:
            Dictionary with SHAP values and summary
        """
        if not SHAP_AVAILABLE:
            raise InvalidParameterError("SHAP not installed. Install with: pip install shap")

        self.logger.info("Calculating SHAP values...")

        # Sample training data for background
        if len(X_train) > sample_size:
            background = X_train.sample(n=sample_size, random_state=42)
        else:
            background = X_train

        # Create explainer based on model type
        try:
            # Try TreeExplainer for tree-based models
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                explainer = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer for other models
                explainer = shap.KernelExplainer(model.predict, background)

            self.explainers['shap'] = explainer

            # Calculate SHAP values for test set (sample if large)
            if len(X_test) > sample_size:
                X_explain = X_test.sample(n=sample_size, random_state=42)
            else:
                X_explain = X_test

            shap_values = explainer.shap_values(X_explain)

            # Handle multi-class case (SHAP values is a list)
            if isinstance(shap_values, list):
                # For multiclass, take the first class for simplicity
                shap_values_data = shap_values[0]
            else:
                shap_values_data = shap_values

            # Calculate mean absolute SHAP values (global importance)
            mean_abs_shap = np.abs(shap_values_data).mean(axis=0)
            feature_importance = dict(zip(X_test.columns, mean_abs_shap))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

            result = {
                "shap_values": shap_values_data.tolist(),
                "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else None,
                "feature_names": list(X_test.columns),
                "global_importance": feature_importance,
                "n_samples_explained": len(X_explain)
            }

            return result

        except Exception as e:
            self.logger.error(f"SHAP calculation failed: {e}")
            raise

    def explain_instance_lime(
        self,
        model: Any,
        X_train: pd.DataFrame,
        instance: pd.Series,
        feature_names: List[str],
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain single prediction using LIME

        Args:
            model: Trained model
            X_train: Training data
            instance: Single instance to explain
            feature_names: Feature names
            num_features: Number of top features to show

        Returns:
            LIME explanation dictionary
        """
        if not LIME_AVAILABLE:
            raise InvalidParameterError("LIME not installed. Install with: pip install lime")

        self.logger.info("Generating LIME explanation...")

        # Determine if classification or regression
        is_classification = hasattr(model, 'predict_proba')

        # Create LIME explainer
        if is_classification:
            explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                mode='classification',
                random_state=42
            )
        else:
            explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                mode='regression',
                random_state=42
            )

        self.explainers['lime'] = explainer

        # Generate explanation
        if is_classification:
            explanation = explainer.explain_instance(
                instance.values,
                model.predict_proba,
                num_features=num_features
            )
        else:
            explanation = explainer.explain_instance(
                instance.values,
                model.predict,
                num_features=num_features
            )

        # Extract explanation data
        exp_list = explanation.as_list()

        result = {
            "feature_contributions": {feat: weight for feat, weight in exp_list},
            "predicted_value": float(model.predict([instance.values])[0]),
            "instance_values": instance.to_dict()
        }

        if is_classification:
            proba = model.predict_proba([instance.values])[0]
            result["predicted_probabilities"] = {f"class_{i}": float(p) for i, p in enumerate(proba)}

        return result

    def partial_dependence_analysis(
        self,
        model: Any,
        X: pd.DataFrame,
        features: List[Union[int, str]],
        grid_resolution: int = 50
    ) -> Dict[str, Any]:
        """
        Calculate partial dependence for features

        Args:
            model: Trained model
            X: Feature data
            features: Features to analyze (indices or names)
            grid_resolution: Number of grid points

        Returns:
            Partial dependence results
        """
        self.logger.info("Calculating partial dependence...")

        # Convert feature names to indices if needed
        feature_indices = []
        for feat in features:
            if isinstance(feat, str):
                feature_indices.append(X.columns.get_loc(feat))
            else:
                feature_indices.append(feat)

        # Calculate partial dependence
        pdp_results = {}

        for idx, feat_idx in enumerate(feature_indices):
            feat_name = X.columns[feat_idx] if isinstance(X, pd.DataFrame) else f"feature_{feat_idx}"

            try:
                pdp = partial_dependence(
                    model, X, [feat_idx], grid_resolution=grid_resolution
                )

                pdp_results[feat_name] = {
                    "values": pdp['grid_values'][0].tolist(),
                    "average": pdp['average'][0].tolist()
                }
            except Exception as e:
                self.logger.warning(f"Could not calculate PDP for {feat_name}: {e}")

        return pdp_results

    def explain_prediction(
        self,
        model: Any,
        X_train: pd.DataFrame,
        instance: pd.Series,
        method: str = 'shap'
    ) -> Dict[str, Any]:
        """
        Explain a single prediction

        Args:
            model: Trained model
            X_train: Training data (for baseline)
            instance: Single instance to explain
            method: 'shap' or 'lime'

        Returns:
            Explanation for the prediction
        """
        prediction = model.predict([instance.values])[0]

        explanation = {
            "prediction": float(prediction),
            "instance": instance.to_dict()
        }

        if method == 'shap' and SHAP_AVAILABLE:
            # Create SHAP explainer if not exists
            if 'shap' not in self.explainers:
                self.calculate_shap_values(model, X_train, pd.DataFrame([instance]))

            explainer = self.explainers['shap']
            shap_values = explainer.shap_values(instance.values.reshape(1, -1))

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            explanation['method'] = 'shap'
            explanation['feature_contributions'] = dict(zip(instance.index, shap_values[0]))

        elif method == 'lime' and LIME_AVAILABLE:
            lime_exp = self.explain_instance_lime(model, X_train, instance, list(instance.index))
            explanation['method'] = 'lime'
            explanation.update(lime_exp)

        return explanation

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            class_names: Class names

        Returns:
            Evaluation metrics and analysis
        """
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            roc_auc_score, log_loss, cohen_kappa_score
        )

        results = {}

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        results['accuracy'] = float(accuracy)
        results['precision'] = float(precision)
        results['recall'] = float(recall)
        results['f1_score'] = float(f1)

        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        results['cohen_kappa'] = float(kappa)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        results['classification_report'] = report

        # ROC AUC and Log Loss (if probabilities provided)
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    # Binary classification
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class
                    roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr')

                results['roc_auc'] = float(roc_auc)

                # Log loss
                logloss = log_loss(y_true, y_proba)
                results['log_loss'] = float(logloss)
            except Exception as e:
                self.logger.debug(f"Could not calculate ROC AUC or log loss: {e}")

        return results

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive regression evaluation

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Evaluation metrics and analysis
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            mean_absolute_percentage_error, explained_variance_score
        )

        results = {}

        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)

        results['mse'] = float(mse)
        results['rmse'] = float(rmse)
        results['mae'] = float(mae)
        results['r2'] = float(r2)
        results['explained_variance'] = float(explained_var)

        # MAPE
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
            results['mape'] = float(mape)
        except:
            pass

        # Residual analysis
        residuals = y_true - y_pred
        results['residuals'] = {
            'mean': float(residuals.mean()),
            'std': float(residuals.std()),
            'min': float(residuals.min()),
            'max': float(residuals.max()),
            'q25': float(np.percentile(residuals, 25)),
            'q50': float(np.percentile(residuals, 50)),
            'q75': float(np.percentile(residuals, 75))
        }

        return results

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Compare multiple models

        Args:
            models: Dictionary of {name: model}
            X_test: Test features
            y_test: Test target
            task_type: 'classification' or 'regression'

        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing {len(models)} models...")

        results = {}

        for name, model in models.items():
            y_pred = model.predict(X_test)

            if task_type == 'classification':
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                metrics = self.evaluate_classification(y_test.values, y_pred, y_proba)
            else:
                metrics = self.evaluate_regression(y_test.values, y_pred)

            results[name] = metrics

        return results
