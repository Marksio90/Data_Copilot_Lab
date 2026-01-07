"""
Data Copilot Lab - Base Classes for Data Cleaning
Abstract base classes for data cleaning operations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.exceptions import DataCleaningError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner(ABC):
    """
    Abstract base class for all data cleaning operations

    All concrete cleaners must implement the clean method
    """

    def __init__(self):
        self.logger = logger
        self._original_data: Optional[pd.DataFrame] = None
        self._cleaned_data: Optional[pd.DataFrame] = None
        self._cleaning_report: Dict[str, Any] = {}

    @abstractmethod
    def clean(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Clean the data

        Args:
            data: DataFrame to clean
            **kwargs: Cleaner-specific parameters

        Returns:
            Cleaned DataFrame

        Raises:
            DataCleaningError: If cleaning fails
        """
        pass

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data quality issues without cleaning

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary with analysis results
        """
        pass

    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Get report of cleaning operations performed

        Returns:
            Dictionary with cleaning statistics
        """
        return self._cleaning_report

    def get_comparison(self) -> Dict[str, Any]:
        """
        Compare original and cleaned data

        Returns:
            Dictionary with before/after comparison
        """
        if self._original_data is None or self._cleaned_data is None:
            raise DataCleaningError("No cleaning operation has been performed")

        return {
            "original": {
                "rows": len(self._original_data),
                "columns": len(self._original_data.columns),
                "missing_values": self._original_data.isnull().sum().sum(),
            },
            "cleaned": {
                "rows": len(self._cleaned_data),
                "columns": len(self._cleaned_data.columns),
                "missing_values": self._cleaned_data.isnull().sum().sum(),
            },
            "changes": {
                "rows_removed": len(self._original_data) - len(self._cleaned_data),
                "columns_removed": len(self._original_data.columns) - len(self._cleaned_data.columns),
                "missing_values_handled": (
                    self._original_data.isnull().sum().sum() -
                    self._cleaned_data.isnull().sum().sum()
                ),
            }
        }

    def _store_original(self, data: pd.DataFrame):
        """Store a copy of original data for comparison"""
        self._original_data = data.copy()

    def _store_cleaned(self, data: pd.DataFrame):
        """Store cleaned data"""
        self._cleaned_data = data.copy()


class CleaningStrategy(ABC):
    """
    Abstract base class for cleaning strategies

    Strategies define HOW to clean (e.g., how to handle missing values)
    """

    @abstractmethod
    def apply(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply cleaning strategy to data

        Args:
            data: DataFrame to clean
            **kwargs: Strategy-specific parameters

        Returns:
            Cleaned DataFrame
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get strategy description"""
        pass


class CleaningPipeline:
    """
    Pipeline for chaining multiple cleaning operations

    Allows building complex cleaning workflows from individual cleaners
    """

    def __init__(self, name: str = "Cleaning Pipeline"):
        self.name = name
        self._steps: List[tuple] = []  # List of (name, cleaner, params)
        self.logger = logger

    def add_step(
        self,
        name: str,
        cleaner: DataCleaner,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Add a cleaning step to the pipeline

        Args:
            name: Name of the step
            cleaner: DataCleaner instance
            params: Parameters for the cleaner
        """
        self._steps.append((name, cleaner, params or {}))
        self.logger.info(f"Added step '{name}' to pipeline")

    def remove_step(self, name: str):
        """
        Remove a step from the pipeline

        Args:
            name: Name of step to remove
        """
        self._steps = [step for step in self._steps if step[0] != name]
        self.logger.info(f"Removed step '{name}' from pipeline")

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the entire pipeline

        Args:
            data: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Executing pipeline '{self.name}' with {len(self._steps)} steps")

        result = data.copy()
        reports = []

        for i, (step_name, cleaner, params) in enumerate(self._steps, 1):
            self.logger.info(f"Step {i}/{len(self._steps)}: {step_name}")

            try:
                result = cleaner.clean(result, **params)
                report = cleaner.get_cleaning_report()
                reports.append({
                    "step": step_name,
                    "report": report
                })

            except Exception as e:
                self.logger.error(f"Error in step '{step_name}': {e}", exc_info=True)
                raise DataCleaningError(
                    f"Pipeline failed at step '{step_name}': {str(e)}"
                )

        self.logger.info(f"Pipeline '{self.name}' completed successfully")

        # Store combined report
        self._pipeline_report = {
            "pipeline_name": self.name,
            "total_steps": len(self._steps),
            "step_reports": reports
        }

        return result

    def get_pipeline_report(self) -> Dict[str, Any]:
        """Get full pipeline execution report"""
        if not hasattr(self, '_pipeline_report'):
            return {"error": "Pipeline has not been executed yet"}
        return self._pipeline_report

    def get_steps(self) -> List[str]:
        """Get list of step names in the pipeline"""
        return [step[0] for step in self._steps]

    def clear(self):
        """Remove all steps from the pipeline"""
        self._steps = []
        self.logger.info(f"Cleared all steps from pipeline '{self.name}'")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize pipeline to dictionary

        Returns:
            Dictionary representation of pipeline
        """
        return {
            "name": self.name,
            "steps": [
                {
                    "name": step[0],
                    "cleaner_type": step[1].__class__.__name__,
                    "params": step[2]
                }
                for step in self._steps
            ]
        }


class DataQualityAnalyzer:
    """
    Comprehensive data quality analysis

    Combines multiple analysis methods to provide overall quality assessment
    """

    def __init__(self):
        self.logger = logger

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality analysis

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary with quality metrics
        """
        self.logger.info("Performing data quality analysis...")

        analysis = {
            "basic_info": self._analyze_basic_info(data),
            "missing_data": self._analyze_missing_data(data),
            "duplicates": self._analyze_duplicates(data),
            "data_types": self._analyze_data_types(data),
            "outliers": self._analyze_outliers(data),
            "quality_score": 0.0  # Will be calculated
        }

        # Calculate overall quality score (0-100)
        analysis["quality_score"] = self._calculate_quality_score(analysis)

        return analysis

    def _analyze_basic_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic dataset information"""
        return {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "columns": list(data.columns)
        }

    def _analyze_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data"""
        missing_counts = data.isnull().sum()
        total_cells = len(data) * len(data.columns)
        total_missing = missing_counts.sum()

        return {
            "total_missing": int(total_missing),
            "missing_percentage": (total_missing / total_cells * 100) if total_cells > 0 else 0,
            "columns_with_missing": missing_counts[missing_counts > 0].to_dict(),
            "rows_with_any_missing": int(data.isnull().any(axis=1).sum())
        }

    def _analyze_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows"""
        n_duplicates = data.duplicated().sum()

        return {
            "n_duplicates": int(n_duplicates),
            "duplicate_percentage": (n_duplicates / len(data) * 100) if len(data) > 0 else 0
        }

    def _analyze_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types"""
        dtype_counts = data.dtypes.value_counts()

        return {
            "type_distribution": dtype_counts.to_dict(),
            "numeric_columns": list(data.select_dtypes(include=['number']).columns),
            "categorical_columns": list(data.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": list(data.select_dtypes(include=['datetime']).columns)
        }

    def _analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential outliers in numeric columns"""
        numeric_cols = data.select_dtypes(include=['number']).columns
        outlier_info = {}

        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]

            outlier_info[col] = {
                "n_outliers": len(outliers),
                "outlier_percentage": (len(outliers) / len(data) * 100) if len(data) > 0 else 0
            }

        return {
            "columns_analyzed": len(numeric_cols),
            "outlier_info": outlier_info
        }

    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate overall quality score (0-100)

        Higher score = better quality
        """
        score = 100.0

        # Deduct for missing data (max -30 points)
        missing_pct = analysis["missing_data"]["missing_percentage"]
        score -= min(missing_pct * 3, 30)

        # Deduct for duplicates (max -20 points)
        duplicate_pct = analysis["duplicates"]["duplicate_percentage"]
        score -= min(duplicate_pct * 2, 20)

        # Deduct for outliers (max -20 points)
        avg_outlier_pct = 0
        outlier_info = analysis["outliers"]["outlier_info"]
        if outlier_info:
            avg_outlier_pct = sum(
                info["outlier_percentage"] for info in outlier_info.values()
            ) / len(outlier_info)
        score -= min(avg_outlier_pct * 2, 20)

        return max(0.0, score)
