"""
Data Copilot Lab - CSV/TSV Importer
Import data from CSV and TSV files with automatic detection
"""

import csv
from pathlib import Path
from typing import List, Optional, Union

import chardet
import pandas as pd

from src.core.config import settings
from src.core.exceptions import DataImportError, DataParsingError
from src.modules.data_import.base import FileImporter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CSVImporter(FileImporter):
    """
    CSV/TSV file importer with automatic detection of:
    - Encoding (UTF-8, Latin-1, etc.)
    - Delimiter (comma, tab, semicolon, etc.)
    - Header presence
    - Data types
    """

    SUPPORTED_EXTENSIONS = ['.csv', '.tsv', '.txt']
    COMMON_DELIMITERS = [',', '\t', ';', '|', ' ']

    def __init__(self):
        super().__init__()
        self._encoding: Optional[str] = None
        self._delimiter: Optional[str] = None
        self._has_header: bool = True

    def import_data(
        self,
        source: Union[str, Path],
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
        has_header: bool = True,
        skip_rows: Optional[int] = None,
        nrows: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Import data from CSV/TSV file

        Args:
            source: Path to CSV file
            delimiter: Column delimiter (auto-detected if None)
            encoding: File encoding (auto-detected if None)
            has_header: Whether first row is header
            skip_rows: Number of rows to skip at start
            nrows: Number of rows to read (None = all)
            **kwargs: Additional pandas.read_csv parameters

        Returns:
            DataFrame with imported data

        Raises:
            DataImportError: If file cannot be read
            DataParsingError: If data cannot be parsed
        """
        try:
            # Validate source
            file_path = Path(source)
            if not self.validate_source(file_path):
                raise DataImportError(f"Invalid file source: {file_path}")

            self._file_path = file_path

            # Check file extension
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                logger.warning(
                    f"File extension {file_path.suffix} not in {self.SUPPORTED_EXTENSIONS}. "
                    "Attempting to import anyway..."
                )

            # Auto-detect encoding if not provided
            if encoding is None:
                encoding = self._detect_encoding(file_path)
                logger.info(f"Detected encoding: {encoding}")
            self._encoding = encoding

            # Auto-detect delimiter if not provided
            if delimiter is None:
                delimiter = self._detect_delimiter(file_path, encoding)
                logger.info(f"Detected delimiter: {repr(delimiter)}")
            self._delimiter = delimiter

            self._has_header = has_header

            # Import data with pandas
            logger.info(f"Importing CSV file: {file_path}")

            read_params = {
                'filepath_or_buffer': file_path,
                'delimiter': delimiter,
                'encoding': encoding,
                'header': 0 if has_header else None,
                'skiprows': skip_rows,
                'nrows': nrows,
                'engine': 'python',  # More flexible for edge cases
                'on_bad_lines': 'warn',  # Warn about bad lines but continue
            }

            # Add any additional kwargs
            read_params.update(kwargs)

            self._data = pd.read_csv(**read_params)

            # Generate column names if no header
            if not has_header:
                self._data.columns = [f"Column_{i}" for i in range(len(self._data.columns))]

            # Store metadata
            self._metadata = {
                'source_type': 'csv',
                'source_path': str(file_path.absolute()),
                'encoding': encoding,
                'delimiter': delimiter,
                'has_header': has_header,
                'rows_imported': len(self._data),
                'columns_imported': len(self._data.columns),
            }

            logger.info(
                f"Successfully imported {len(self._data)} rows "
                f"and {len(self._data.columns)} columns"
            )

            return self._data

        except pd.errors.EmptyDataError:
            raise DataImportError(f"File is empty: {source}")

        except pd.errors.ParserError as e:
            raise DataParsingError(f"Error parsing CSV file: {str(e)}")

        except Exception as e:
            logger.error(f"Error importing CSV file: {e}", exc_info=True)
            raise DataImportError(f"Failed to import CSV file: {str(e)}")

    def _detect_encoding(self, file_path: Path, sample_size: int = 100000) -> str:
        """
        Auto-detect file encoding using chardet

        Args:
            file_path: Path to file
            sample_size: Number of bytes to sample

        Returns:
            Detected encoding (e.g., 'utf-8', 'latin-1')
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']

                logger.debug(
                    f"Encoding detection: {encoding} "
                    f"(confidence: {confidence:.2%})"
                )

                # Fallback to utf-8 if confidence is too low
                if confidence < 0.5:
                    logger.warning(
                        f"Low encoding confidence ({confidence:.2%}), "
                        "falling back to utf-8"
                    )
                    return 'utf-8'

                return encoding

        except Exception as e:
            logger.warning(f"Error detecting encoding: {e}. Using utf-8")
            return 'utf-8'

    def _detect_delimiter(
        self,
        file_path: Path,
        encoding: str,
        sample_lines: int = 10
    ) -> str:
        """
        Auto-detect CSV delimiter

        Args:
            file_path: Path to file
            encoding: File encoding
            sample_lines: Number of lines to sample

        Returns:
            Detected delimiter
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Read sample lines
                sample = []
                for i, line in enumerate(f):
                    if i >= sample_lines:
                        break
                    sample.append(line)

                sample_text = ''.join(sample)

            # Use csv.Sniffer to detect delimiter
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample_text).delimiter

            logger.debug(f"CSV Sniffer detected delimiter: {repr(delimiter)}")

            # Validate delimiter is in common delimiters
            if delimiter not in self.COMMON_DELIMITERS:
                logger.warning(
                    f"Unusual delimiter detected: {repr(delimiter)}. "
                    "Falling back to comma."
                )
                return ','

            return delimiter

        except Exception as e:
            logger.warning(
                f"Error detecting delimiter: {e}. "
                f"Falling back to comma."
            )
            return ','

    def get_column_types(self) -> dict:
        """
        Get detected column types

        Returns:
            Dictionary mapping column names to their data types
        """
        if self._data is None:
            raise DataImportError("No data has been imported yet")

        return {col: str(dtype) for col, dtype in self._data.dtypes.items()}

    def suggest_type_conversions(self) -> dict:
        """
        Suggest better type conversions for columns

        Returns:
            Dictionary with suggested type conversions
        """
        if self._data is None:
            raise DataImportError("No data has been imported yet")

        suggestions = {}

        for col in self._data.columns:
            current_type = str(self._data[col].dtype)

            # Skip if already optimal type
            if current_type in ['int64', 'float64', 'bool', 'datetime64[ns]']:
                continue

            # Try to infer better type
            if current_type == 'object':
                # Try numeric
                try:
                    pd.to_numeric(self._data[col], errors='raise')
                    suggestions[col] = 'numeric'
                    continue
                except (ValueError, TypeError):
                    pass

                # Try datetime
                try:
                    pd.to_datetime(self._data[col], errors='raise')
                    suggestions[col] = 'datetime'
                    continue
                except (ValueError, TypeError):
                    pass

                # Try boolean
                unique_values = self._data[col].dropna().unique()
                if len(unique_values) <= 2:
                    suggestions[col] = 'boolean'
                    continue

        return suggestions

    def apply_type_conversions(self, conversions: dict) -> pd.DataFrame:
        """
        Apply type conversions to DataFrame

        Args:
            conversions: Dictionary mapping column names to target types
                        (e.g., {'col1': 'numeric', 'col2': 'datetime'})

        Returns:
            DataFrame with converted types
        """
        if self._data is None:
            raise DataImportError("No data has been imported yet")

        for col, target_type in conversions.items():
            if col not in self._data.columns:
                logger.warning(f"Column {col} not found in data. Skipping.")
                continue

            try:
                if target_type == 'numeric':
                    self._data[col] = pd.to_numeric(self._data[col], errors='coerce')
                elif target_type == 'datetime':
                    self._data[col] = pd.to_datetime(self._data[col], errors='coerce')
                elif target_type == 'boolean':
                    self._data[col] = self._data[col].astype('bool')
                else:
                    logger.warning(f"Unknown target type: {target_type}")

                logger.info(f"Converted {col} to {target_type}")

            except Exception as e:
                logger.error(f"Error converting {col} to {target_type}: {e}")

        return self._data
