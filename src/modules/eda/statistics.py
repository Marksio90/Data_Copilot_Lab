"""
Data Copilot Lab - Statistical Analysis
Comprehensive statistical analysis and descriptive statistics
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.core.exceptions import DataCleaningError, InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis tool

    Provides descriptive statistics, distribution analysis,
    hypothesis testing, and statistical summaries
    """

    def __init__(self):
        self.logger = logger

    def describe(self, data: pd.DataFrame, percentiles: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive descriptive statistics

        Args:
            data: DataFrame to analyze
            percentiles: Custom percentiles to include

        Returns:
            Dictionary with statistics for all columns
        """
        self.logger.info("Generating descriptive statistics...")

        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]

        numeric_stats = self._describe_numeric(data, percentiles)
        categorical_stats = self._describe_categorical(data)
        datetime_stats = self._describe_datetime(data)

        return {
            "overview": {
                "n_rows": len(data),
                "n_columns": len(data.columns),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            "numeric": numeric_stats,
            "categorical": categorical_stats,
            "datetime": datetime_stats
        }

    def _describe_numeric(self, data: pd.DataFrame, percentiles: List[float]) -> Dict[str, Any]:
        """Describe numeric columns"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return {"columns": [], "statistics": {}}

        stats_dict = {}

        for col in numeric_cols:
            series = data[col].dropna()

            if len(series) == 0:
                continue

            stats_dict[col] = {
                "count": len(series),
                "missing": data[col].isnull().sum(),
                "missing_pct": (data[col].isnull().sum() / len(data) * 100),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "range": float(series.max() - series.min()),
                "percentiles": {
                    f"p{int(p*100)}": float(series.quantile(p))
                    for p in percentiles
                },
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "variance": float(series.var()),
                "sum": float(series.sum()),
                "median": float(series.median()),
                "mode": float(series.mode()[0]) if not series.mode().empty else None,
                "unique_values": int(series.nunique()),
                "coefficient_of_variation": float(series.std() / series.mean()) if series.mean() != 0 else None
            }

        return {
            "columns": numeric_cols,
            "count": len(numeric_cols),
            "statistics": stats_dict
        }

    def _describe_categorical(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Describe categorical columns"""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            return {"columns": [], "statistics": {}}

        stats_dict = {}

        for col in categorical_cols:
            series = data[col].dropna()

            if len(series) == 0:
                continue

            value_counts = series.value_counts()
            mode_value = value_counts.index[0] if len(value_counts) > 0 else None

            stats_dict[col] = {
                "count": len(series),
                "missing": data[col].isnull().sum(),
                "missing_pct": (data[col].isnull().sum() / len(data) * 100),
                "unique_values": int(series.nunique()),
                "mode": str(mode_value),
                "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "mode_percentage": float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0,
                "top_5_values": value_counts.head(5).to_dict(),
                "cardinality": "high" if series.nunique() > 50 else "medium" if series.nunique() > 10 else "low"
            }

        return {
            "columns": categorical_cols,
            "count": len(categorical_cols),
            "statistics": stats_dict
        }

    def _describe_datetime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Describe datetime columns"""
        datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()

        if not datetime_cols:
            return {"columns": [], "statistics": {}}

        stats_dict = {}

        for col in datetime_cols:
            series = data[col].dropna()

            if len(series) == 0:
                continue

            stats_dict[col] = {
                "count": len(series),
                "missing": data[col].isnull().sum(),
                "missing_pct": (data[col].isnull().sum() / len(data) * 100),
                "min": str(series.min()),
                "max": str(series.max()),
                "range": str(series.max() - series.min()),
                "unique_values": int(series.nunique())
            }

        return {
            "columns": datetime_cols,
            "count": len(datetime_cols),
            "statistics": stats_dict
        }

    def distribution_analysis(self, data: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Analyze distribution of a numeric column

        Args:
            data: DataFrame
            column: Column to analyze

        Returns:
            Distribution analysis results
        """
        if column not in data.columns:
            raise InvalidParameterError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(data[column]):
            raise InvalidParameterError(f"Column '{column}' must be numeric")

        series = data[column].dropna()

        if len(series) < 3:
            raise DataCleaningError(f"Not enough data points in column '{column}'")

        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(series[:5000])  # Shapiro-Wilk (max 5000 samples)

        # Anderson-Darling test
        anderson_result = stats.anderson(series)

        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(series)

        # Distribution parameters
        mean = series.mean()
        std = series.std()
        skew = series.skew()
        kurt = series.kurtosis()

        # Determine distribution type
        is_normal = shapiro_p > 0.05
        is_uniform = self._test_uniformity(series)
        is_exponential = self._test_exponential(series)

        return {
            "column": column,
            "distribution_tests": {
                "shapiro_wilk": {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": bool(shapiro_p > 0.05)
                },
                "anderson_darling": {
                    "statistic": float(anderson_result.statistic),
                    "critical_values": anderson_result.critical_values.tolist(),
                    "significance_levels": anderson_result.significance_level.tolist()
                },
                "jarque_bera": {
                    "statistic": float(jb_stat),
                    "p_value": float(jb_p),
                    "is_normal": bool(jb_p > 0.05)
                }
            },
            "parameters": {
                "mean": float(mean),
                "std": float(std),
                "skewness": float(skew),
                "kurtosis": float(kurt)
            },
            "distribution_type": {
                "normal": is_normal,
                "uniform": is_uniform,
                "exponential": is_exponential,
                "suggested": self._suggest_distribution(is_normal, is_uniform, is_exponential, skew)
            }
        }

    def _test_uniformity(self, series: pd.Series) -> bool:
        """Test if distribution is approximately uniform"""
        # Simple test: check if std is small relative to range
        data_range = series.max() - series.min()
        expected_std = data_range / np.sqrt(12)  # Theoretical std for uniform distribution
        actual_std = series.std()

        return abs(actual_std - expected_std) / expected_std < 0.2

    def _test_exponential(self, series: pd.Series) -> bool:
        """Test if distribution is approximately exponential"""
        # Exponential should have high positive skewness
        skew = series.skew()
        return skew > 1.5

    def _suggest_distribution(self, normal: bool, uniform: bool, exponential: bool, skew: float) -> str:
        """Suggest best-fitting distribution"""
        if normal:
            return "normal"
        elif uniform:
            return "uniform"
        elif exponential:
            return "exponential"
        elif abs(skew) > 1:
            return "log-normal" if skew > 0 else "negatively_skewed"
        else:
            return "unknown"

    def hypothesis_test(
        self,
        data1: pd.Series,
        data2: Optional[pd.Series] = None,
        test_type: str = "t-test",
        alternative: str = "two-sided"
    ) -> Dict[str, Any]:
        """
        Perform hypothesis testing

        Args:
            data1: First sample
            data2: Second sample (for two-sample tests)
            test_type: Type of test ('t-test', 'mann-whitney', 'ks-test', 'chi-square')
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Test results
        """
        data1 = data1.dropna()

        if test_type == "t-test":
            if data2 is None:
                # One-sample t-test
                stat, pval = stats.ttest_1samp(data1, 0)
                test_name = "One-sample t-test"
            else:
                # Two-sample t-test
                data2 = data2.dropna()
                stat, pval = stats.ttest_ind(data1, data2, alternative=alternative)
                test_name = "Two-sample t-test"

        elif test_type == "mann-whitney":
            if data2 is None:
                raise InvalidParameterError("Mann-Whitney test requires two samples")
            data2 = data2.dropna()
            stat, pval = stats.mannwhitneyu(data1, data2, alternative=alternative)
            test_name = "Mann-Whitney U test"

        elif test_type == "ks-test":
            if data2 is None:
                # One-sample KS test (against normal distribution)
                stat, pval = stats.kstest(data1, 'norm')
                test_name = "Kolmogorov-Smirnov test (vs normal)"
            else:
                # Two-sample KS test
                data2 = data2.dropna()
                stat, pval = stats.ks_2samp(data1, data2, alternative=alternative)
                test_name = "Kolmogorov-Smirnov test (two-sample)"

        elif test_type == "chi-square":
            if data2 is None:
                raise InvalidParameterError("Chi-square test requires two samples")
            # Chi-square test for independence
            contingency_table = pd.crosstab(data1, data2)
            stat, pval, dof, expected = stats.chi2_contingency(contingency_table)
            test_name = "Chi-square test of independence"

        else:
            raise InvalidParameterError(f"Unknown test type: {test_type}")

        return {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(pval),
            "significant": bool(pval < 0.05),
            "alternative": alternative
        }

    def summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate quick summary statistics for the entire dataset

        Returns:
            Summary dictionary
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        return {
            "dataset_info": {
                "n_rows": len(data),
                "n_columns": len(data.columns),
                "n_numeric": len(numeric_cols),
                "n_categorical": len(categorical_cols),
                "memory_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            "data_quality": {
                "total_cells": len(data) * len(data.columns),
                "missing_cells": data.isnull().sum().sum(),
                "missing_percentage": (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100),
                "duplicate_rows": data.duplicated().sum(),
                "duplicate_percentage": (data.duplicated().sum() / len(data) * 100)
            },
            "numeric_summary": {
                "mean": data[numeric_cols].mean().to_dict() if len(numeric_cols) > 0 else {},
                "std": data[numeric_cols].std().to_dict() if len(numeric_cols) > 0 else {},
                "median": data[numeric_cols].median().to_dict() if len(numeric_cols) > 0 else {}
            },
            "categorical_summary": {
                col: {
                    "unique_values": data[col].nunique(),
                    "top_value": data[col].mode()[0] if not data[col].mode().empty else None
                }
                for col in categorical_cols
            }
        }

    def percentile_analysis(
        self,
        data: pd.DataFrame,
        column: str,
        percentiles: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Detailed percentile analysis for a column

        Args:
            data: DataFrame
            column: Column to analyze
            percentiles: List of percentiles (0-1)

        Returns:
            Percentile analysis
        """
        if column not in data.columns:
            raise InvalidParameterError(f"Column '{column}' not found")

        if not pd.api.types.is_numeric_dtype(data[column]):
            raise InvalidParameterError(f"Column '{column}' must be numeric")

        if percentiles is None:
            percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        series = data[column].dropna()

        percentile_values = {
            f"p{int(p*100)}": float(series.quantile(p))
            for p in percentiles
        }

        # IQR analysis
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25

        return {
            "column": column,
            "percentiles": percentile_values,
            "iqr_analysis": {
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(iqr),
                "lower_fence": float(q25 - 1.5 * iqr),
                "upper_fence": float(q75 + 1.5 * iqr)
            },
            "deciles": {
                f"d{i}": float(series.quantile(i/10))
                for i in range(11)
            }
        }

    def frequency_analysis(self, data: pd.DataFrame, column: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Frequency analysis for categorical columns

        Args:
            data: DataFrame
            column: Column to analyze
            top_n: Number of top values to return

        Returns:
            Frequency analysis
        """
        if column not in data.columns:
            raise InvalidParameterError(f"Column '{column}' not found")

        series = data[column]
        value_counts = series.value_counts()
        value_percentages = series.value_counts(normalize=True) * 100

        return {
            "column": column,
            "total_count": len(series),
            "unique_values": series.nunique(),
            "missing_values": series.isnull().sum(),
            "top_values": {
                str(val): {
                    "count": int(value_counts[val]),
                    "percentage": float(value_percentages[val])
                }
                for val in value_counts.head(top_n).index
            },
            "entropy": float(stats.entropy(value_counts))  # Shannon entropy
        }
