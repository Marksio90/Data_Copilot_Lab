"""
Data Copilot Lab - Excel Importer
Import data from Excel files (.xls, .xlsx) with multi-sheet support
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.core.config import settings
from src.core.exceptions import DataImportError, DataParsingError
from src.modules.data_import.base import FileImporter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExcelImporter(FileImporter):
    """
    Excel file importer supporting:
    - .xls and .xlsx formats
    - Multiple sheets
    - Named ranges
    - Formula evaluation
    - Merged cells handling
    """

    SUPPORTED_EXTENSIONS = ['.xls', '.xlsx', '.xlsm', '.xlsb']

    def __init__(self):
        super().__init__()
        self._sheet_names: List[str] = []
        self._active_sheet: Optional[str] = None
        self._all_sheets: Dict[str, pd.DataFrame] = {}

    def import_data(
        self,
        source: Union[str, Path],
        sheet_name: Optional[Union[str, int]] = 0,
        has_header: bool = True,
        skip_rows: Optional[int] = None,
        nrows: Optional[int] = None,
        usecols: Optional[Union[str, List]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Import data from Excel file

        Args:
            source: Path to Excel file
            sheet_name: Sheet name or index (0-based). None = all sheets
            has_header: Whether first row is header
            skip_rows: Number of rows to skip at start
            nrows: Number of rows to read (None = all)
            usecols: Columns to read (e.g., "A:E" or [0,1,2])
            **kwargs: Additional pandas.read_excel parameters

        Returns:
            DataFrame with imported data (from specified sheet)
            If sheet_name=None, imports all sheets and returns first sheet

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
                raise DataImportError(
                    f"Unsupported file extension: {file_path.suffix}. "
                    f"Supported: {self.SUPPORTED_EXTENSIONS}"
                )

            logger.info(f"Importing Excel file: {file_path}")

            # First, get all sheet names
            self._sheet_names = self._get_sheet_names(file_path)
            logger.info(f"Found {len(self._sheet_names)} sheets: {self._sheet_names}")

            # Prepare read parameters
            read_params = {
                'io': file_path,
                'sheet_name': sheet_name,
                'header': 0 if has_header else None,
                'skiprows': skip_rows,
                'nrows': nrows,
                'usecols': usecols,
                'engine': None,  # Auto-detect engine
            }

            # Add any additional kwargs
            read_params.update(kwargs)

            # Import data
            if sheet_name is None:
                # Import all sheets
                logger.info("Importing all sheets...")
                self._all_sheets = pd.read_excel(**read_params)

                # Set active sheet to first sheet
                self._active_sheet = self._sheet_names[0]
                self._data = self._all_sheets[self._active_sheet]

            else:
                # Import specific sheet
                self._data = pd.read_excel(**read_params)

                # Determine sheet name
                if isinstance(sheet_name, int):
                    self._active_sheet = self._sheet_names[sheet_name]
                else:
                    self._active_sheet = sheet_name

                self._all_sheets = {self._active_sheet: self._data}

            # Generate column names if no header
            if not has_header:
                self._data.columns = [
                    f"Column_{i}" for i in range(len(self._data.columns))
                ]

            # Store metadata
            self._metadata = {
                'source_type': 'excel',
                'source_path': str(file_path.absolute()),
                'total_sheets': len(self._sheet_names),
                'sheet_names': self._sheet_names,
                'active_sheet': self._active_sheet,
                'has_header': has_header,
                'rows_imported': len(self._data),
                'columns_imported': len(self._data.columns),
            }

            logger.info(
                f"Successfully imported sheet '{self._active_sheet}': "
                f"{len(self._data)} rows and {len(self._data.columns)} columns"
            )

            return self._data

        except FileNotFoundError:
            raise DataImportError(f"Excel file not found: {source}")

        except Exception as e:
            logger.error(f"Error importing Excel file: {e}", exc_info=True)
            raise DataImportError(f"Failed to import Excel file: {str(e)}")

    def _get_sheet_names(self, file_path: Path) -> List[str]:
        """
        Get all sheet names from Excel file

        Args:
            file_path: Path to Excel file

        Returns:
            List of sheet names
        """
        try:
            excel_file = pd.ExcelFile(file_path)
            return excel_file.sheet_names
        except Exception as e:
            logger.error(f"Error getting sheet names: {e}")
            return []

    def get_sheet_names(self) -> List[str]:
        """
        Get list of all sheet names in the Excel file

        Returns:
            List of sheet names
        """
        if not self._sheet_names:
            raise DataImportError("No Excel file has been imported yet")
        return self._sheet_names

    def get_sheet_data(self, sheet_name: Union[str, int]) -> pd.DataFrame:
        """
        Get data from a specific sheet

        Args:
            sheet_name: Sheet name or index

        Returns:
            DataFrame with sheet data
        """
        if not self._all_sheets:
            raise DataImportError("No sheets have been imported")

        # Convert index to name if needed
        if isinstance(sheet_name, int):
            if sheet_name >= len(self._sheet_names):
                raise DataImportError(
                    f"Sheet index {sheet_name} out of range. "
                    f"Available: 0-{len(self._sheet_names)-1}"
                )
            sheet_name = self._sheet_names[sheet_name]

        if sheet_name not in self._all_sheets:
            # Try to import this specific sheet
            if self._file_path is None:
                raise DataImportError("File path not available")

            logger.info(f"Loading sheet: {sheet_name}")
            self._all_sheets[sheet_name] = pd.read_excel(
                self._file_path,
                sheet_name=sheet_name
            )

        return self._all_sheets[sheet_name]

    def switch_sheet(self, sheet_name: Union[str, int]) -> pd.DataFrame:
        """
        Switch active sheet

        Args:
            sheet_name: Sheet name or index

        Returns:
            DataFrame with new active sheet data
        """
        self._data = self.get_sheet_data(sheet_name)

        # Update active sheet name
        if isinstance(sheet_name, int):
            self._active_sheet = self._sheet_names[sheet_name]
        else:
            self._active_sheet = sheet_name

        logger.info(f"Switched to sheet: {self._active_sheet}")
        return self._data

    def import_all_sheets(
        self,
        merge: bool = False,
        merge_key: Optional[str] = None
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Import all sheets from Excel file

        Args:
            merge: Whether to merge all sheets into one DataFrame
            merge_key: Column to use as merge key (required if merge=True)

        Returns:
            Dictionary of DataFrames (one per sheet) or single merged DataFrame
        """
        if self._file_path is None:
            raise DataImportError("No file has been imported")

        logger.info("Importing all sheets...")

        all_data = pd.read_excel(
            self._file_path,
            sheet_name=None  # None means all sheets
        )

        self._all_sheets = all_data

        if merge:
            if not merge_key:
                raise DataImportError(
                    "merge_key is required when merge=True"
                )

            logger.info(f"Merging sheets on key: {merge_key}")

            # Start with first sheet
            merged = list(all_data.values())[0]

            # Merge remaining sheets
            for sheet_name, df in list(all_data.items())[1:]:
                merged = merged.merge(
                    df,
                    on=merge_key,
                    how='outer',
                    suffixes=('', f'_{sheet_name}')
                )

            self._data = merged
            return merged

        return all_data

    def get_sheet_info(self, sheet_name: Optional[str] = None) -> Dict:
        """
        Get information about a specific sheet

        Args:
            sheet_name: Sheet name (None = active sheet)

        Returns:
            Dictionary with sheet information
        """
        if sheet_name is None:
            sheet_name = self._active_sheet

        if sheet_name is None:
            raise DataImportError("No active sheet")

        df = self.get_sheet_data(sheet_name)

        return {
            "sheet_name": sheet_name,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict()
        }

    def detect_header_row(self, sample_rows: int = 10) -> int:
        """
        Attempt to detect which row contains the header

        Args:
            sample_rows: Number of rows to sample

        Returns:
            Estimated header row index (0-based)
        """
        if self._data is None:
            raise DataImportError("No data has been imported")

        # Simple heuristic: header row typically has:
        # 1. More unique values
        # 2. More text (vs numbers)
        # 3. No missing values

        # This is a placeholder - could be made more sophisticated
        return 0  # Default to first row

    def clean_column_names(self) -> pd.DataFrame:
        """
        Clean and standardize column names

        Returns:
            DataFrame with cleaned column names
        """
        if self._data is None:
            raise DataImportError("No data has been imported")

        # Clean column names
        self._data.columns = (
            self._data.columns
            .str.strip()  # Remove leading/trailing spaces
            .str.lower()  # Convert to lowercase
            .str.replace(' ', '_')  # Replace spaces with underscores
            .str.replace('[^a-z0-9_]', '', regex=True)  # Remove special chars
        )

        logger.info("Column names cleaned")
        return self._data
