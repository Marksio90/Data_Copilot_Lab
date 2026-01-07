"""
Data Copilot Lab - Missing Data Handler
Detect and handle missing values with multiple strategies
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from src.core.exceptions import DataCleaningError, InvalidParameterError
from src.modules.data_cleaning.base import DataCleaner, CleaningStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MissingValueStrategy(str, Enum):
    """Strategies for handling missing values"""
    DROP_ROWS = "drop_rows"
    DROP_COLUMNS = "drop_columns"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_CONSTANT = "fill_constant"
    FILL_FORWARD = "fill_forward"
    FILL_BACKWARD = "fill_backward"
    FILL_INTERPOLATE = "fill_interpolate"
    FILL_KNN = "fill_knn"
    FILL_ITERATIVE = "fill_iterative"


class MissingDataHandler(DataCleaner):
    """
    Handle missing data with various strategies

    Supports multiple imputation methods:
    - Statistical (mean, median, mode)
    - Forward/backward fill
    - Interpolation
    - Advanced (KNN, Iterative)
    - Drop rows/columns
    """

    def __init__(self):
        super().__init__()

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing data patterns

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary with missing data analysis
        """
        self.logger.info("Analyzing missing data...")

        missing_counts = data.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        total_cells = len(data) * len(data.columns)
        total_missing = missing_counts.sum()

        # Missing data patterns
        missing_patterns = []
        for col in missing_cols.index:
            missing_patterns.append({
                "column": col,
                "missing_count": int(missing_cols[col]),
                "missing_percentage": (missing_cols[col] / len(data) * 100),
                "dtype": str(data[col].dtype),
                "suggested_strategy": self._suggest_strategy(data[col])
            })

        # Rows with missing data
        rows_with_missing = data.isnull().any(axis=1).sum()

        # Correlation of missingness (which columns tend to have missing values together)
        missing_corr = data.isnull().corr()

        return {
            "total_cells": total_cells,
            "total_missing": int(total_missing),
            "missing_percentage": (total_missing / total_cells * 100) if total_cells > 0 else 0,
            "columns_with_missing": len(missing_cols),
            "rows_with_missing": int(rows_with_missing),
            "missing_patterns": missing_patterns,
            "missingness_correlation": missing_corr.to_dict() if len(missing_cols) > 1 else {},
            "recommendations": self._generate_recommendations(data, missing_patterns)
        }

    def clean(
        self,
        data: pd.DataFrame,
        strategy: Union[str, MissingValueStrategy] = MissingValueStrategy.FILL_MEAN,
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        fill_value: Any = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Clean missing data

        Args:
            data: DataFrame to clean
            strategy: Strategy to use for handling missing values
            columns: Specific columns to clean (None = all columns)
            threshold: For drop strategies, threshold of missing % to drop
            fill_value: For FILL_CONSTANT strategy
            **kwargs: Additional strategy-specific parameters

        Returns:
            Cleaned DataFrame
        """
        self._store_original(data)
        result = data.copy()

        # Convert string strategy to enum
        if isinstance(strategy, str):
            try:
                strategy = MissingValueStrategy(strategy)
            except ValueError:
                raise InvalidParameterError(
                    f"Invalid strategy: {strategy}. "
                    f"Valid strategies: {[s.value for s in MissingValueStrategy]}"
                )

        self.logger.info(f"Cleaning missing data with strategy: {strategy.value}")

        # Determine columns to process
        if columns is None:
            # Find columns with missing values
            columns = data.columns[data.isnull().any()].tolist()
        else:
            # Validate columns exist
            invalid_cols = set(columns) - set(data.columns)
            if invalid_cols:
                raise InvalidParameterError(f"Columns not found: {invalid_cols}")

        if not columns:
            self.logger.info("No missing values found. Nothing to clean.")
            self._store_cleaned(result)
            self._cleaning_report = {"message": "No missing values found"}
            return result

        # Apply strategy
        if strategy == MissingValueStrategy.DROP_ROWS:
            result = self._drop_rows(result, columns, threshold)

        elif strategy == MissingValueStrategy.DROP_COLUMNS:
            result = self._drop_columns(result, columns, threshold)

        elif strategy == MissingValueStrategy.FILL_MEAN:
            result = self._fill_mean(result, columns)

        elif strategy == MissingValueStrategy.FILL_MEDIAN:
            result = self._fill_median(result, columns)

        elif strategy == MissingValueStrategy.FILL_MODE:
            result = self._fill_mode(result, columns)

        elif strategy == MissingValueStrategy.FILL_CONSTANT:
            if fill_value is None:
                raise InvalidParameterError("fill_value required for FILL_CONSTANT strategy")
            result = self._fill_constant(result, columns, fill_value)

        elif strategy == MissingValueStrategy.FILL_FORWARD:
            result = self._fill_forward(result, columns)

        elif strategy == MissingValueStrategy.FILL_BACKWARD:
            result = self._fill_backward(result, columns)

        elif strategy == MissingValueStrategy.FILL_INTERPOLATE:
            result = self._fill_interpolate(result, columns, **kwargs)

        elif strategy == MissingValueStrategy.FILL_KNN:
            result = self._fill_knn(result, columns, **kwargs)

        elif strategy == MissingValueStrategy.FILL_ITERATIVE:
            result = self._fill_iterative(result, columns, **kwargs)

        self._store_cleaned(result)

        # Generate cleaning report
        self._cleaning_report = {
            "strategy": strategy.value,
            "columns_processed": len(columns),
            "original_missing": int(data.isnull().sum().sum()),
            "cleaned_missing": int(result.isnull().sum().sum()),
            "missing_handled": int(data.isnull().sum().sum() - result.isnull().sum().sum()),
            "rows_before": len(data),
            "rows_after": len(result),
            "rows_removed": len(data) - len(result)
        }

        self.logger.info(f"Missing data handling complete. Handled {self._cleaning_report['missing_handled']} missing values")

        return result

    def _drop_rows(
        self,
        data: pd.DataFrame,
        columns: List[str],
        threshold: Optional[float]
    ) -> pd.DataFrame:
        """Drop rows with missing values"""
        if threshold:
            # Drop rows where % of missing values exceeds threshold
            missing_pct = data[columns].isnull().sum(axis=1) / len(columns)
            result = data[missing_pct <= threshold/100]
        else:
            # Drop any row with missing values in specified columns
            result = data.dropna(subset=columns)

        return result

    def _drop_columns(
        self,
        data: pd.DataFrame,
        columns: List[str],
        threshold: Optional[float]
    ) -> pd.DataFrame:
        """Drop columns with excessive missing values"""
        result = data.copy()

        if threshold:
            # Drop columns where % of missing exceeds threshold
            for col in columns:
                missing_pct = data[col].isnull().sum() / len(data) * 100
                if missing_pct > threshold:
                    result = result.drop(columns=[col])
                    self.logger.info(f"Dropped column '{col}' ({missing_pct:.1f}% missing)")
        else:
            # Drop all columns with any missing values
            result = result.drop(columns=columns)

        return result

    def _fill_mean(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values with mean (numeric columns only)"""
        result = data.copy()

        for col in columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                mean_value = result[col].mean()
                result[col].fillna(mean_value, inplace=True)
                self.logger.debug(f"Filled '{col}' with mean: {mean_value}")
            else:
                self.logger.warning(f"Skipping non-numeric column '{col}' for mean fill")

        return result

    def _fill_median(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values with median (numeric columns only)"""
        result = data.copy()

        for col in columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                median_value = result[col].median()
                result[col].fillna(median_value, inplace=True)
                self.logger.debug(f"Filled '{col}' with median: {median_value}")
            else:
                self.logger.warning(f"Skipping non-numeric column '{col}' for median fill")

        return result

    def _fill_mode(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values with mode (most frequent value)"""
        result = data.copy()

        for col in columns:
            if not result[col].empty:
                mode_value = result[col].mode()
                if not mode_value.empty:
                    result[col].fillna(mode_value[0], inplace=True)
                    self.logger.debug(f"Filled '{col}' with mode: {mode_value[0]}")

        return result

    def _fill_constant(
        self,
        data: pd.DataFrame,
        columns: List[str],
        value: Any
    ) -> pd.DataFrame:
        """Fill missing values with a constant value"""
        result = data.copy()

        for col in columns:
            result[col].fillna(value, inplace=True)
            self.logger.debug(f"Filled '{col}' with constant: {value}")

        return result

    def _fill_forward(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Forward fill - propagate last valid observation forward"""
        result = data.copy()

        for col in columns:
            result[col].fillna(method='ffill', inplace=True)
            self.logger.debug(f"Forward filled '{col}'")

        return result

    def _fill_backward(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Backward fill - use next valid observation to fill gap"""
        result = data.copy()

        for col in columns:
            result[col].fillna(method='bfill', inplace=True)
            self.logger.debug(f"Backward filled '{col}'")

        return result

    def _fill_interpolate(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'linear',
        **kwargs
    ) -> pd.DataFrame:
        """Fill missing values using interpolation"""
        result = data.copy()

        for col in columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].interpolate(method=method, **kwargs)
                self.logger.debug(f"Interpolated '{col}' with method: {method}")
            else:
                self.logger.warning(f"Skipping non-numeric column '{col}' for interpolation")

        return result

    def _fill_knn(
        self,
        data: pd.DataFrame,
        columns: List[str],
        n_neighbors: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        """Fill missing values using K-Nearest Neighbors"""
        result = data.copy()

        # KNN imputer works on numeric data only
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(result[col])]

        if not numeric_cols:
            self.logger.warning("No numeric columns for KNN imputation")
            return result

        imputer = KNNImputer(n_neighbors=n_neighbors, **kwargs)
        result[numeric_cols] = imputer.fit_transform(result[numeric_cols])

        self.logger.info(f"KNN imputation applied to {len(numeric_cols)} columns")

        return result

    def _fill_iterative(
        self,
        data: pd.DataFrame,
        columns: List[str],
        max_iter: int = 10,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fill missing values using Iterative Imputer (MICE algorithm)

        Models each feature with missing values as a function of other features
        """
        result = data.copy()

        # Iterative imputer works on numeric data only
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(result[col])]

        if not numeric_cols:
            self.logger.warning("No numeric columns for iterative imputation")
            return result

        imputer = IterativeImputer(max_iter=max_iter, random_state=42, **kwargs)
        result[numeric_cols] = imputer.fit_transform(result[numeric_cols])

        self.logger.info(f"Iterative imputation applied to {len(numeric_cols)} columns")

        return result

    def _suggest_strategy(self, series: pd.Series) -> str:
        """Suggest best strategy for a column based on data type and missing %"""
        missing_pct = series.isnull().sum() / len(series) * 100

        # If too many missing values, suggest dropping
        if missing_pct > 50:
            return "drop_columns"

        # For numeric data
        if pd.api.types.is_numeric_dtype(series):
            if missing_pct < 5:
                return "fill_median"  # Median is robust to outliers
            elif missing_pct < 20:
                return "fill_knn"  # KNN for moderate missingness
            else:
                return "fill_iterative"  # Iterative for higher missingness

        # For categorical data
        else:
            if missing_pct < 10:
                return "fill_mode"
            else:
                return "fill_constant"  # Or drop

    def _generate_recommendations(
        self,
        data: pd.DataFrame,
        patterns: List[Dict]
    ) -> List[str]:
        """Generate human-readable recommendations"""
        recommendations = []

        # Check overall missing percentage
        total_missing_pct = sum(p["missing_percentage"] for p in patterns) / len(patterns) if patterns else 0

        if total_missing_pct > 30:
            recommendations.append(
                "⚠️ High overall missing data (>30%). Consider data collection improvements."
            )

        # Check for columns with extreme missingness
        extreme_cols = [p for p in patterns if p["missing_percentage"] > 70]
        if extreme_cols:
            cols_str = ", ".join([p["column"] for p in extreme_cols])
            recommendations.append(
                f"🗑️ Consider dropping columns with >70% missing: {cols_str}"
            )

        # Check for patterns
        if len(patterns) > 5:
            recommendations.append(
                "📊 Many columns have missing values. Consider using advanced imputation (KNN, Iterative)."
            )

        if not recommendations:
            recommendations.append("✅ Missing data is manageable. Apply suggested strategies per column.")

        return recommendations

    def visualize_missing_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data for visualizing missing patterns

        Returns dictionary with visualization data (frontend can render this)
        """
        missing_matrix = data.isnull().astype(int)

        return {
            "matrix": missing_matrix.values.tolist(),
            "columns": list(data.columns),
            "rows": len(data),
            "missing_per_column": data.isnull().sum().to_dict(),
            "missing_per_row": missing_matrix.sum(axis=1).tolist()
        }
