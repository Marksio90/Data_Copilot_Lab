"""
Data Copilot Lab - Data Standardizer
Standardize and normalize data formats, types, and encodings
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder
)

from src.core.exceptions import DataCleaningError, InvalidParameterError
from src.modules.data_cleaning.base import DataCleaner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScalingMethod(str, Enum):
    """Methods for scaling numeric data"""
    STANDARD = "standard"  # Z-score normalization
    MINMAX = "minmax"  # Scale to [0, 1]
    ROBUST = "robust"  # Robust to outliers
    LOG = "log"  # Log transformation
    SQRT = "sqrt"  # Square root transformation


class EncodingMethod(str, Enum):
    """Methods for encoding categorical data"""
    LABEL = "label"  # Label encoding (ordinal)
    ONEHOT = "onehot"  # One-hot encoding
    ORDINAL = "ordinal"  # Ordinal encoding with custom order
    BINARY = "binary"  # Binary encoding
    FREQUENCY = "frequency"  # Frequency encoding


class DataStandardizer(DataCleaner):
    """
    Standardize data formats, types, and encodings

    Features:
    - Numeric scaling (standardization, normalization)
    - Categorical encoding (label, one-hot)
    - Date/time formatting
    - String cleaning and formatting
    - Unit conversions
    """

    def __init__(self):
        super().__init__()
        self._scalers: Dict[str, Any] = {}
        self._encoders: Dict[str, Any] = {}

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data standardization opportunities

        Args:
            data: DataFrame to analyze

        Returns:
            Analysis results with suggestions
        """
        self.logger.info("Analyzing data standardization opportunities...")

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()

        analysis = {
            "numeric_columns": {
                "count": len(numeric_cols),
                "columns": numeric_cols,
                "needs_scaling": self._check_scaling_needed(data, numeric_cols)
            },
            "categorical_columns": {
                "count": len(categorical_cols),
                "columns": categorical_cols,
                "needs_encoding": self._check_encoding_needed(data, categorical_cols),
                "high_cardinality": [
                    col for col in categorical_cols
                    if data[col].nunique() > 50
                ]
            },
            "datetime_columns": {
                "count": len(datetime_cols),
                "columns": datetime_cols
            },
            "recommendations": self._generate_standardization_recommendations(
                data, numeric_cols, categorical_cols
            )
        }

        return analysis

    def clean(
        self,
        data: pd.DataFrame,
        scale_numeric: bool = True,
        scaling_method: Union[str, ScalingMethod] = ScalingMethod.STANDARD,
        encode_categorical: bool = True,
        encoding_method: Union[str, EncodingMethod] = EncodingMethod.LABEL,
        standardize_strings: bool = True,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Standardize data

        Args:
            data: DataFrame to clean
            scale_numeric: Whether to scale numeric columns
            scaling_method: Method for scaling
            encode_categorical: Whether to encode categorical columns
            encoding_method: Method for encoding
            standardize_strings: Whether to clean string columns
            numeric_columns: Specific numeric columns to scale
            categorical_columns: Specific categorical columns to encode

        Returns:
            Standardized DataFrame
        """
        self._store_original(data)
        result = data.copy()

        operations_performed = []

        # Scale numeric columns
        if scale_numeric:
            if isinstance(scaling_method, str):
                scaling_method = ScalingMethod(scaling_method)

            result = self.scale_numeric(
                result,
                method=scaling_method,
                columns=numeric_columns
            )
            operations_performed.append(f"scaled_numeric_{scaling_method.value}")

        # Encode categorical columns
        if encode_categorical:
            if isinstance(encoding_method, str):
                encoding_method = EncodingMethod(encoding_method)

            result = self.encode_categorical(
                result,
                method=encoding_method,
                columns=categorical_columns
            )
            operations_performed.append(f"encoded_categorical_{encoding_method.value}")

        # Standardize strings
        if standardize_strings:
            result = self.standardize_strings(result)
            operations_performed.append("standardized_strings")

        self._store_cleaned(result)

        # Generate report
        self._cleaning_report = {
            "operations": operations_performed,
            "rows_processed": len(result),
            "columns_before": len(data.columns),
            "columns_after": len(result.columns),
            "columns_added": len(result.columns) - len(data.columns)
        }

        return result

    def scale_numeric(
        self,
        data: pd.DataFrame,
        method: ScalingMethod = ScalingMethod.STANDARD,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Scale numeric columns

        Args:
            data: DataFrame
            method: Scaling method
            columns: Specific columns to scale (None = all numeric)

        Returns:
            DataFrame with scaled columns
        """
        result = data.copy()

        # Determine columns
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Validate columns are numeric
            for col in columns:
                if col not in result.columns:
                    raise InvalidParameterError(f"Column '{col}' not found")
                if not pd.api.types.is_numeric_dtype(result[col]):
                    raise InvalidParameterError(f"Column '{col}' is not numeric")

        if not columns:
            self.logger.info("No numeric columns to scale")
            return result

        self.logger.info(f"Scaling {len(columns)} numeric columns with method: {method.value}")

        for col in columns:
            if method == ScalingMethod.STANDARD:
                scaler = StandardScaler()
                result[col] = scaler.fit_transform(result[[col]])
                self._scalers[col] = scaler

            elif method == ScalingMethod.MINMAX:
                scaler = MinMaxScaler()
                result[col] = scaler.fit_transform(result[[col]])
                self._scalers[col] = scaler

            elif method == ScalingMethod.ROBUST:
                scaler = RobustScaler()
                result[col] = scaler.fit_transform(result[[col]])
                self._scalers[col] = scaler

            elif method == ScalingMethod.LOG:
                # Handle negative values
                min_val = result[col].min()
                if min_val <= 0:
                    result[col] = result[col] - min_val + 1
                result[col] = np.log1p(result[col])

            elif method == ScalingMethod.SQRT:
                # Handle negative values
                min_val = result[col].min()
                if min_val < 0:
                    result[col] = result[col] - min_val
                result[col] = np.sqrt(result[col])

        return result

    def encode_categorical(
        self,
        data: pd.DataFrame,
        method: EncodingMethod = EncodingMethod.LABEL,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Encode categorical columns

        Args:
            data: DataFrame
            method: Encoding method
            columns: Specific columns to encode (None = all categorical)

        Returns:
            DataFrame with encoded columns
        """
        result = data.copy()

        # Determine columns
        if columns is None:
            columns = result.select_dtypes(include=['object', 'category']).columns.tolist()

        if not columns:
            self.logger.info("No categorical columns to encode")
            return result

        self.logger.info(f"Encoding {len(columns)} categorical columns with method: {method.value}")

        for col in columns:
            if col not in result.columns:
                continue

            if method == EncodingMethod.LABEL:
                encoder = LabelEncoder()
                result[col] = encoder.fit_transform(result[col].astype(str))
                self._encoders[col] = encoder

            elif method == EncodingMethod.ONEHOT:
                # One-hot encoding creates new columns
                dummies = pd.get_dummies(result[col], prefix=col, drop_first=False)
                result = pd.concat([result, dummies], axis=1)
                result = result.drop(columns=[col])

            elif method == EncodingMethod.FREQUENCY:
                # Encode by frequency
                freq_map = result[col].value_counts(normalize=True).to_dict()
                result[col] = result[col].map(freq_map)

            elif method == EncodingMethod.BINARY:
                # Binary encoding (custom implementation)
                unique_values = result[col].unique()
                if len(unique_values) == 2:
                    result[col] = (result[col] == unique_values[0]).astype(int)
                else:
                    self.logger.warning(f"Binary encoding works best with 2 unique values. Column '{col}' has {len(unique_values)}")

        return result

    def standardize_strings(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lowercase: bool = True,
        strip: bool = True,
        remove_special: bool = False
    ) -> pd.DataFrame:
        """
        Standardize string columns

        Args:
            data: DataFrame
            columns: Specific columns (None = all string columns)
            lowercase: Convert to lowercase
            strip: Remove leading/trailing whitespace
            remove_special: Remove special characters

        Returns:
            DataFrame with standardized strings
        """
        result = data.copy()

        if columns is None:
            columns = result.select_dtypes(include=['object']).columns.tolist()

        for col in columns:
            if col not in result.columns:
                continue

            if lowercase:
                result[col] = result[col].astype(str).str.lower()

            if strip:
                result[col] = result[col].astype(str).str.strip()

            if remove_special:
                result[col] = result[col].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

        self.logger.info(f"Standardized {len(columns)} string columns")

        return result

    def standardize_dates(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        format: str = "%Y-%m-%d"
    ) -> pd.DataFrame:
        """
        Standardize date columns to consistent format

        Args:
            data: DataFrame
            columns: Date columns to standardize
            format: Target date format

        Returns:
            DataFrame with standardized dates
        """
        result = data.copy()

        if columns is None:
            columns = result.select_dtypes(include=['datetime']).columns.tolist()

        for col in columns:
            if col not in result.columns:
                continue

            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(result[col]):
                result[col] = pd.to_datetime(result[col], errors='coerce')

            # Format as string
            result[col] = result[col].dt.strftime(format)

        self.logger.info(f"Standardized {len(columns)} date columns")

        return result

    def normalize_column_names(
        self,
        data: pd.DataFrame,
        lowercase: bool = True,
        replace_spaces: bool = True,
        remove_special: bool = True
    ) -> pd.DataFrame:
        """
        Normalize column names

        Args:
            data: DataFrame
            lowercase: Convert to lowercase
            replace_spaces: Replace spaces with underscores
            remove_special: Remove special characters

        Returns:
            DataFrame with normalized column names
        """
        result = data.copy()

        new_columns = []
        for col in result.columns:
            new_col = str(col)

            if lowercase:
                new_col = new_col.lower()

            if replace_spaces:
                new_col = new_col.replace(' ', '_')

            if remove_special:
                import re
                new_col = re.sub(r'[^a-z0-9_]', '', new_col)

            new_columns.append(new_col)

        result.columns = new_columns

        self.logger.info("Normalized column names")

        return result

    def _check_scaling_needed(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, bool]:
        """Check which numeric columns need scaling"""
        needs_scaling = {}

        for col in columns:
            if col not in data.columns:
                continue

            # Check if values are already normalized (mean~0, std~1)
            mean = data[col].mean()
            std = data[col].std()
            min_val = data[col].min()
            max_val = data[col].max()

            # Needs scaling if not already normalized or in [0,1] range
            if not (-0.5 < mean < 0.5 and 0.8 < std < 1.2):
                if not (0 <= min_val and max_val <= 1):
                    needs_scaling[col] = True
                else:
                    needs_scaling[col] = False
            else:
                needs_scaling[col] = False

        return needs_scaling

    def _check_encoding_needed(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, bool]:
        """Check which categorical columns need encoding"""
        needs_encoding = {}

        for col in columns:
            if col not in data.columns:
                continue

            # Check if already numeric
            try:
                pd.to_numeric(data[col])
                needs_encoding[col] = False
            except (ValueError, TypeError):
                needs_encoding[col] = True

        return needs_encoding

    def _generate_standardization_recommendations(
        self,
        data: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> List[str]:
        """Generate recommendations for standardization"""
        recommendations = []

        # Check numeric scaling
        if numeric_cols:
            needs_scaling = self._check_scaling_needed(data, numeric_cols)
            cols_needing_scaling = [col for col, needs in needs_scaling.items() if needs]

            if cols_needing_scaling:
                recommendations.append(
                    f"📊 Scale numeric columns ({len(cols_needing_scaling)}): "
                    f"{', '.join(cols_needing_scaling[:3])}"
                    f"{' and more...' if len(cols_needing_scaling) > 3 else ''}"
                )

        # Check categorical encoding
        if categorical_cols:
            high_cardinality = [
                col for col in categorical_cols
                if data[col].nunique() > 50
            ]

            if high_cardinality:
                recommendations.append(
                    f"⚠️ High cardinality columns ({len(high_cardinality)}): "
                    f"Consider frequency encoding or dropping"
                )

            if len(categorical_cols) - len(high_cardinality) > 0:
                recommendations.append(
                    f"🔤 Encode categorical columns: Use label encoding for ordinal, "
                    f"one-hot for nominal"
                )

        # Check column names
        has_special = any(not col.replace('_', '').isalnum() for col in data.columns)
        if has_special:
            recommendations.append(
                "📝 Normalize column names: Remove special characters"
            )

        if not recommendations:
            recommendations.append("✅ Data is well-standardized")

        return recommendations

    def get_scaling_info(self, column: str) -> Optional[Dict[str, Any]]:
        """Get scaler information for a column"""
        if column in self._scalers:
            scaler = self._scalers[column]
            return {
                "column": column,
                "scaler_type": type(scaler).__name__,
                "mean": getattr(scaler, 'mean_', None),
                "scale": getattr(scaler, 'scale_', None)
            }
        return None

    def get_encoding_info(self, column: str) -> Optional[Dict[str, Any]]:
        """Get encoder information for a column"""
        if column in self._encoders:
            encoder = self._encoders[column]
            return {
                "column": column,
                "encoder_type": type(encoder).__name__,
                "classes": getattr(encoder, 'classes_', None).tolist() if hasattr(encoder, 'classes_') else None
            }
        return None
