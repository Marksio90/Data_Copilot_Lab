"""
Data Copilot Lab - Outlier Detector
Detect and handle outliers using statistical and ML methods
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from src.core.exceptions import DataCleaningError, InvalidParameterError
from src.modules.data_cleaning.base import DataCleaner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OutlierMethod(str, Enum):
    """Methods for detecting outliers"""
    IQR = "iqr"  # Interquartile Range
    ZSCORE = "zscore"  # Z-score (standard deviations from mean)
    MODIFIED_ZSCORE = "modified_zscore"  # Modified Z-score (using median)
    ISOLATION_FOREST = "isolation_forest"  # ML-based
    LOF = "lof"  # Local Outlier Factor
    ELLIPTIC_ENVELOPE = "elliptic_envelope"  # Gaussian distribution assumption


class OutlierAction(str, Enum):
    """Actions to take on outliers"""
    REMOVE = "remove"  # Remove outlier rows
    CAP = "cap"  # Cap at threshold
    CLIP = "clip"  # Clip to percentile range
    WINSORIZE = "winsorize"  # Replace with percentile values
    LOG_TRANSFORM = "log_transform"  # Log transformation to reduce impact
    MARK = "mark"  # Just mark/flag outliers without removing


class OutlierDetector(DataCleaner):
    """
    Detect and handle outliers using multiple methods

    Supports:
    - Statistical methods (IQR, Z-score)
    - Machine learning methods (Isolation Forest, LOF)
    - Multiple handling strategies (remove, cap, transform)
    """

    def __init__(self):
        super().__init__()

    def analyze(self, data: pd.DataFrame, method: Union[str, OutlierMethod] = OutlierMethod.IQR) -> Dict[str, Any]:
        """
        Analyze outliers in the data

        Args:
            data: DataFrame to analyze
            method: Detection method to use

        Returns:
            Dictionary with outlier analysis
        """
        self.logger.info(f"Analyzing outliers using method: {method}")

        # Convert string to enum
        if isinstance(method, str):
            method = OutlierMethod(method)

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return {
                "message": "No numeric columns found for outlier detection",
                "outliers_found": False
            }

        outliers_per_column = {}
        total_outliers = 0

        for col in numeric_cols:
            outlier_indices = self._detect_outliers_column(
                data[col],
                method=method
            )

            outliers_per_column[col] = {
                "n_outliers": len(outlier_indices),
                "outlier_percentage": (len(outlier_indices) / len(data) * 100),
                "outlier_values": data.loc[outlier_indices, col].tolist()[:10]  # First 10 for preview
            }

            total_outliers += len(outlier_indices)

        return {
            "method": method.value,
            "columns_analyzed": len(numeric_cols),
            "total_outliers": total_outliers,
            "outlier_percentage": (total_outliers / (len(data) * len(numeric_cols)) * 100) if len(data) > 0 else 0,
            "outliers_per_column": outliers_per_column,
            "recommendations": self._generate_recommendations(outliers_per_column)
        }

    def clean(
        self,
        data: pd.DataFrame,
        method: Union[str, OutlierMethod] = OutlierMethod.IQR,
        action: Union[str, OutlierAction] = OutlierAction.REMOVE,
        columns: Optional[List[str]] = None,
        threshold: float = 1.5,
        **kwargs
    ) -> pd.DataFrame:
        """
        Detect and handle outliers

        Args:
            data: DataFrame to clean
            method: Detection method
            action: Action to take on outliers
            columns: Specific columns to process (None = all numeric)
            threshold: Threshold for detection (meaning varies by method)
            **kwargs: Method-specific parameters

        Returns:
            Cleaned DataFrame
        """
        self._store_original(data)
        result = data.copy()

        # Convert strings to enums
        if isinstance(method, str):
            method = OutlierMethod(method)
        if isinstance(action, str):
            action = OutlierAction(action)

        self.logger.info(f"Detecting outliers: method={method.value}, action={action.value}")

        # Determine columns to process
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Validate columns
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            invalid_cols = set(columns) - set(numeric_cols)
            if invalid_cols:
                raise InvalidParameterError(
                    f"Columns must be numeric for outlier detection: {invalid_cols}"
                )

        if not columns:
            self.logger.info("No numeric columns found. Nothing to clean.")
            self._store_cleaned(result)
            self._cleaning_report = {"message": "No numeric columns found"}
            return result

        outlier_counts = {}

        # Process each column
        for col in columns:
            outlier_indices = self._detect_outliers_column(
                result[col],
                method=method,
                threshold=threshold,
                **kwargs
            )

            outlier_counts[col] = len(outlier_indices)

            if len(outlier_indices) == 0:
                continue

            # Apply action
            if action == OutlierAction.REMOVE:
                # Mark for removal (we'll remove all at once later)
                result.loc[outlier_indices, '_outlier_flag'] = True

            elif action == OutlierAction.CAP:
                result = self._cap_outliers(result, col, outlier_indices)

            elif action == OutlierAction.CLIP:
                result = self._clip_outliers(result, col, **kwargs)

            elif action == OutlierAction.WINSORIZE:
                result = self._winsorize_outliers(result, col, **kwargs)

            elif action == OutlierAction.LOG_TRANSFORM:
                result = self._log_transform(result, col)

            elif action == OutlierAction.MARK:
                result.loc[outlier_indices, f'{col}_outlier'] = True

        # Remove flagged rows if action was REMOVE
        if action == OutlierAction.REMOVE and '_outlier_flag' in result.columns:
            rows_before = len(result)
            result = result[result['_outlier_flag'] != True]  # noqa
            result = result.drop(columns=['_outlier_flag'])
            rows_removed = rows_before - len(result)
            self.logger.info(f"Removed {rows_removed} rows with outliers")

        self._store_cleaned(result)

        # Generate report
        total_outliers = sum(outlier_counts.values())
        self._cleaning_report = {
            "method": method.value,
            "action": action.value,
            "columns_processed": len(columns),
            "outliers_detected": total_outliers,
            "outliers_per_column": outlier_counts,
            "rows_before": len(data),
            "rows_after": len(result),
            "rows_removed": len(data) - len(result)
        }

        return result

    def _detect_outliers_column(
        self,
        series: pd.Series,
        method: OutlierMethod,
        threshold: float = 1.5,
        **kwargs
    ) -> pd.Index:
        """
        Detect outliers in a single column

        Args:
            series: Column data
            method: Detection method
            threshold: Detection threshold

        Returns:
            Indices of outliers
        """
        if method == OutlierMethod.IQR:
            return self._detect_iqr(series, threshold)

        elif method == OutlierMethod.ZSCORE:
            return self._detect_zscore(series, threshold)

        elif method == OutlierMethod.MODIFIED_ZSCORE:
            return self._detect_modified_zscore(series, threshold)

        elif method == OutlierMethod.ISOLATION_FOREST:
            return self._detect_isolation_forest(series, **kwargs)

        elif method == OutlierMethod.LOF:
            return self._detect_lof(series, **kwargs)

        elif method == OutlierMethod.ELLIPTIC_ENVELOPE:
            return self._detect_elliptic_envelope(series, **kwargs)

        else:
            raise InvalidParameterError(f"Unknown outlier detection method: {method}")

    def _detect_iqr(self, series: pd.Series, threshold: float = 1.5) -> pd.Index:
        """
        Detect outliers using Interquartile Range (IQR) method

        Outliers are values below Q1 - threshold*IQR or above Q3 + threshold*IQR
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = (series < lower_bound) | (series > upper_bound)
        return series[outliers].index

    def _detect_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Index:
        """
        Detect outliers using Z-score method

        Outliers are values with |z-score| > threshold (typically 3)
        """
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = z_scores > threshold

        return series.dropna().iloc[outliers].index

    def _detect_modified_zscore(self, series: pd.Series, threshold: float = 3.5) -> pd.Index:
        """
        Detect outliers using Modified Z-score (more robust to outliers)

        Uses median and MAD (Median Absolute Deviation) instead of mean and std
        """
        median = series.median()
        mad = np.median(np.abs(series - median))

        if mad == 0:
            return pd.Index([])

        modified_z_scores = 0.6745 * (series - median) / mad
        outliers = np.abs(modified_z_scores) > threshold

        return series[outliers].index

    def _detect_isolation_forest(
        self,
        series: pd.Series,
        contamination: float = 0.1,
        **kwargs
    ) -> pd.Index:
        """
        Detect outliers using Isolation Forest (ML method)

        Args:
            contamination: Expected proportion of outliers
        """
        # Isolation Forest requires 2D array
        X = series.dropna().values.reshape(-1, 1)

        clf = IsolationForest(
            contamination=contamination,
            random_state=42,
            **kwargs
        )

        predictions = clf.fit_predict(X)
        outliers = predictions == -1

        return series.dropna().iloc[outliers].index

    def _detect_lof(
        self,
        series: pd.Series,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        **kwargs
    ) -> pd.Index:
        """
        Detect outliers using Local Outlier Factor

        Measures local density deviation
        """
        X = series.dropna().values.reshape(-1, 1)

        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            **kwargs
        )

        predictions = clf.fit_predict(X)
        outliers = predictions == -1

        return series.dropna().iloc[outliers].index

    def _detect_elliptic_envelope(
        self,
        series: pd.Series,
        contamination: float = 0.1,
        **kwargs
    ) -> pd.Index:
        """
        Detect outliers using Elliptic Envelope

        Assumes data is Gaussian and detects outliers based on Mahalanobis distance
        """
        X = series.dropna().values.reshape(-1, 1)

        clf = EllipticEnvelope(
            contamination=contamination,
            random_state=42,
            **kwargs
        )

        predictions = clf.fit_predict(X)
        outliers = predictions == -1

        return series.dropna().iloc[outliers].index

    def _cap_outliers(
        self,
        data: pd.DataFrame,
        column: str,
        outlier_indices: pd.Index
    ) -> pd.DataFrame:
        """Cap outliers at the nearest non-outlier value"""
        result = data.copy()

        # Find bounds (exclude outliers)
        non_outliers = result.loc[~result.index.isin(outlier_indices), column]
        lower_bound = non_outliers.min()
        upper_bound = non_outliers.max()

        # Cap values
        result.loc[outlier_indices, column] = result.loc[outlier_indices, column].clip(
            lower=lower_bound,
            upper=upper_bound
        )

        return result

    def _clip_outliers(
        self,
        data: pd.DataFrame,
        column: str,
        lower_percentile: float = 1,
        upper_percentile: float = 99
    ) -> pd.DataFrame:
        """Clip values to percentile range"""
        result = data.copy()

        lower_bound = result[column].quantile(lower_percentile / 100)
        upper_bound = result[column].quantile(upper_percentile / 100)

        result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)

        return result

    def _winsorize_outliers(
        self,
        data: pd.DataFrame,
        column: str,
        lower_percentile: float = 5,
        upper_percentile: float = 95
    ) -> pd.DataFrame:
        """Winsorize - replace outliers with percentile values"""
        result = data.copy()

        lower_val = result[column].quantile(lower_percentile / 100)
        upper_val = result[column].quantile(upper_percentile / 100)

        # Replace values below lower percentile
        result.loc[result[column] < lower_val, column] = lower_val

        # Replace values above upper percentile
        result.loc[result[column] > upper_val, column] = upper_val

        return result

    def _log_transform(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply log transformation to reduce outlier impact"""
        result = data.copy()

        # Handle negative values
        min_val = result[column].min()
        if min_val <= 0:
            # Shift to make all values positive
            result[column] = result[column] - min_val + 1

        result[column] = np.log1p(result[column])  # log1p = log(1 + x)

        return result

    def _generate_recommendations(self, outliers_per_column: Dict) -> List[str]:
        """Generate recommendations based on outlier analysis"""
        recommendations = []

        # Check columns with high outlier percentage
        high_outlier_cols = [
            col for col, info in outliers_per_column.items()
            if info["outlier_percentage"] > 10
        ]

        if high_outlier_cols:
            recommendations.append(
                f"⚠️ High outlier percentage (>10%) in: {', '.join(high_outlier_cols)}. "
                f"Consider log transformation or investigating data quality."
            )

        # Check if any column has no outliers
        clean_cols = [
            col for col, info in outliers_per_column.items()
            if info["n_outliers"] == 0
        ]

        if clean_cols:
            recommendations.append(
                f"✅ No outliers detected in: {', '.join(clean_cols)}"
            )

        # General advice
        if not recommendations:
            recommendations.append(
                "📊 Moderate outliers detected. Review outliers manually before removing."
            )

        return recommendations

    def compare_methods(
        self,
        data: pd.DataFrame,
        column: str,
        methods: Optional[List[OutlierMethod]] = None
    ) -> Dict[str, Any]:
        """
        Compare different outlier detection methods on a column

        Args:
            data: DataFrame
            column: Column to analyze
            methods: List of methods to compare (None = all methods)

        Returns:
            Comparison results
        """
        if column not in data.columns:
            raise InvalidParameterError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(data[column]):
            raise InvalidParameterError(f"Column '{column}' must be numeric")

        if methods is None:
            methods = list(OutlierMethod)

        comparison = {}

        for method in methods:
            try:
                outlier_indices = self._detect_outliers_column(
                    data[column],
                    method=method
                )

                comparison[method.value] = {
                    "n_outliers": len(outlier_indices),
                    "outlier_percentage": (len(outlier_indices) / len(data) * 100),
                    "outlier_indices": outlier_indices.tolist()[:20]  # First 20
                }

            except Exception as e:
                self.logger.error(f"Error with method {method.value}: {e}")
                comparison[method.value] = {"error": str(e)}

        return {
            "column": column,
            "methods_compared": len(methods),
            "results": comparison
        }
