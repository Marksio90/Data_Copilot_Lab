"""
Data Copilot Lab - JSON/XML Importer
Import data from JSON and XML files with nested structure support
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from bs4 import BeautifulSoup

from src.core.config import settings
from src.core.exceptions import DataImportError, DataParsingError
from src.modules.data_import.base import FileImporter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class JSONImporter(FileImporter):
    """
    JSON file importer supporting:
    - Standard JSON arrays and objects
    - Nested JSON structures
    - JSON Lines (JSONL) format
    - Pretty-printed and minified JSON
    """

    SUPPORTED_EXTENSIONS = ['.json', '.jsonl', '.ndjson']

    def __init__(self):
        super().__init__()
        self._json_data: Optional[Union[Dict, List]] = None
        self._is_nested: bool = False

    def import_data(
        self,
        source: Union[str, Path],
        orient: str = 'records',
        normalize: bool = True,
        max_level: Optional[int] = None,
        record_path: Optional[Union[str, List]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Import data from JSON file

        Args:
            source: Path to JSON file
            orient: JSON orientation ('records', 'index', 'columns', 'values')
            normalize: Whether to normalize nested structures
            max_level: Maximum level to normalize (None = all levels)
            record_path: Path to records in nested JSON
            **kwargs: Additional parameters

        Returns:
            DataFrame with imported data

        Raises:
            DataImportError: If file cannot be read
            DataParsingError: If JSON cannot be parsed
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

            logger.info(f"Importing JSON file: {file_path}")

            # Handle JSON Lines format
            if file_path.suffix.lower() in ['.jsonl', '.ndjson']:
                self._data = self._import_jsonl(file_path)
            else:
                # Load JSON data
                with open(file_path, 'r', encoding='utf-8') as f:
                    self._json_data = json.load(f)

                # Check if nested
                self._is_nested = self._check_if_nested(self._json_data)

                # Convert to DataFrame
                if normalize and self._is_nested:
                    logger.info("Normalizing nested JSON structure...")
                    if record_path:
                        self._data = pd.json_normalize(
                            self._json_data,
                            record_path=record_path,
                            max_level=max_level
                        )
                    else:
                        self._data = pd.json_normalize(
                            self._json_data,
                            max_level=max_level
                        )
                else:
                    self._data = pd.DataFrame(self._json_data)

            # Store metadata
            self._metadata = {
                'source_type': 'json',
                'source_path': str(file_path.absolute()),
                'is_nested': self._is_nested,
                'normalized': normalize,
                'rows_imported': len(self._data),
                'columns_imported': len(self._data.columns),
            }

            logger.info(
                f"Successfully imported {len(self._data)} rows "
                f"and {len(self._data.columns)} columns"
            )

            return self._data

        except json.JSONDecodeError as e:
            raise DataParsingError(f"Invalid JSON format: {str(e)}")

        except Exception as e:
            logger.error(f"Error importing JSON file: {e}", exc_info=True)
            raise DataImportError(f"Failed to import JSON file: {str(e)}")

    def _import_jsonl(self, file_path: Path) -> pd.DataFrame:
        """
        Import JSON Lines format

        Args:
            file_path: Path to JSONL file

        Returns:
            DataFrame with imported data
        """
        records = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping invalid JSON on line {line_num}: {str(e)}"
                    )
                    continue

        return pd.DataFrame(records)

    def _check_if_nested(self, data: Union[Dict, List]) -> bool:
        """
        Check if JSON structure is nested

        Args:
            data: JSON data

        Returns:
            True if nested, False otherwise
        """
        if isinstance(data, list):
            if not data:
                return False
            # Check first element
            return isinstance(data[0], (dict, list))

        elif isinstance(data, dict):
            for value in data.values():
                if isinstance(value, (dict, list)):
                    return True
            return False

        return False

    def flatten_json(
        self,
        sep: str = '_',
        max_level: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Flatten nested JSON structure

        Args:
            sep: Separator for nested keys
            max_level: Maximum nesting level to flatten

        Returns:
            DataFrame with flattened structure
        """
        if self._json_data is None:
            raise DataImportError("No JSON data has been imported")

        def flatten_dict(d: dict, parent_key: str = '', level: int = 0) -> dict:
            """Recursively flatten dictionary"""
            items = []

            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k

                if max_level and level >= max_level:
                    items.append((new_key, v))
                elif isinstance(v, dict):
                    items.extend(
                        flatten_dict(v, new_key, level + 1).items()
                    )
                elif isinstance(v, list):
                    # Convert list to string or handle separately
                    items.append((new_key, str(v)))
                else:
                    items.append((new_key, v))

            return dict(items)

        # Flatten data
        if isinstance(self._json_data, list):
            flattened = [flatten_dict(item) for item in self._json_data]
        else:
            flattened = [flatten_dict(self._json_data)]

        self._data = pd.DataFrame(flattened)
        return self._data

    def get_json_structure(self) -> Dict[str, Any]:
        """
        Get structure information about the JSON

        Returns:
            Dictionary describing JSON structure
        """
        if self._json_data is None:
            raise DataImportError("No JSON data has been imported")

        def analyze_structure(data, path="root"):
            """Recursively analyze JSON structure"""
            if isinstance(data, dict):
                return {
                    "type": "object",
                    "path": path,
                    "keys": list(data.keys()),
                    "children": {
                        k: analyze_structure(v, f"{path}.{k}")
                        for k, v in data.items()
                    }
                }
            elif isinstance(data, list):
                if not data:
                    return {"type": "array", "path": path, "length": 0}
                return {
                    "type": "array",
                    "path": path,
                    "length": len(data),
                    "item_type": analyze_structure(data[0], f"{path}[0]")
                }
            else:
                return {
                    "type": type(data).__name__,
                    "path": path
                }

        return analyze_structure(self._json_data)


class XMLImporter(FileImporter):
    """
    XML file importer supporting:
    - Standard XML structures
    - Nested elements
    - Attributes
    - Multiple record paths
    """

    SUPPORTED_EXTENSIONS = ['.xml']

    def __init__(self):
        super().__init__()
        self._xml_tree: Optional[ET.ElementTree] = None
        self._root: Optional[ET.Element] = None

    def import_data(
        self,
        source: Union[str, Path],
        record_tag: Optional[str] = None,
        include_attributes: bool = True,
        parser: str = 'etree',  # 'etree' or 'beautifulsoup'
        **kwargs
    ) -> pd.DataFrame:
        """
        Import data from XML file

        Args:
            source: Path to XML file
            record_tag: Tag name for records (e.g., 'item', 'row')
            include_attributes: Whether to include XML attributes as columns
            parser: Parser to use ('etree' or 'beautifulsoup')
            **kwargs: Additional parameters

        Returns:
            DataFrame with imported data

        Raises:
            DataImportError: If file cannot be read
            DataParsingError: If XML cannot be parsed
        """
        try:
            # Validate source
            file_path = Path(source)
            if not self.validate_source(file_path):
                raise DataImportError(f"Invalid file source: {file_path}")

            self._file_path = file_path

            logger.info(f"Importing XML file: {file_path}")

            if parser == 'beautifulsoup':
                self._data = self._import_with_beautifulsoup(
                    file_path,
                    record_tag,
                    include_attributes
                )
            else:
                self._data = self._import_with_etree(
                    file_path,
                    record_tag,
                    include_attributes
                )

            # Store metadata
            self._metadata = {
                'source_type': 'xml',
                'source_path': str(file_path.absolute()),
                'record_tag': record_tag,
                'parser': parser,
                'rows_imported': len(self._data),
                'columns_imported': len(self._data.columns),
            }

            logger.info(
                f"Successfully imported {len(self._data)} rows "
                f"and {len(self._data.columns)} columns"
            )

            return self._data

        except ET.ParseError as e:
            raise DataParsingError(f"Invalid XML format: {str(e)}")

        except Exception as e:
            logger.error(f"Error importing XML file: {e}", exc_info=True)
            raise DataImportError(f"Failed to import XML file: {str(e)}")

    def _import_with_etree(
        self,
        file_path: Path,
        record_tag: Optional[str],
        include_attributes: bool
    ) -> pd.DataFrame:
        """Import using ElementTree parser"""
        self._xml_tree = ET.parse(file_path)
        self._root = self._xml_tree.getroot()

        # If no record tag specified, try to detect
        if record_tag is None:
            record_tag = self._detect_record_tag()
            logger.info(f"Detected record tag: {record_tag}")

        # Find all records
        if record_tag:
            records = self._root.findall(f".//{record_tag}")
        else:
            records = [self._root]

        # Parse records
        data = []
        for record in records:
            row = self._parse_element(record, include_attributes)
            data.append(row)

        return pd.DataFrame(data)

    def _import_with_beautifulsoup(
        self,
        file_path: Path,
        record_tag: Optional[str],
        include_attributes: bool
    ) -> pd.DataFrame:
        """Import using BeautifulSoup parser (more forgiving)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')

        # If no record tag specified, try to detect
        if record_tag is None:
            record_tag = self._detect_record_tag_bs(soup)
            logger.info(f"Detected record tag: {record_tag}")

        # Find all records
        if record_tag:
            records = soup.find_all(record_tag)
        else:
            records = [soup]

        # Parse records
        data = []
        for record in records:
            row = self._parse_element_bs(record, include_attributes)
            data.append(row)

        return pd.DataFrame(data)

    def _parse_element(
        self,
        element: ET.Element,
        include_attributes: bool
    ) -> Dict:
        """Parse XML element to dictionary"""
        row = {}

        # Add attributes
        if include_attributes:
            for key, value in element.attrib.items():
                row[f"@{key}"] = value

        # Add child elements
        for child in element:
            tag = child.tag
            # Remove namespace if present
            if '}' in tag:
                tag = tag.split('}')[1]

            # Get text content
            text = child.text.strip() if child.text else None

            # Handle nested elements
            if len(child) > 0:
                # Has children - recursively parse
                row[tag] = self._parse_element(child, include_attributes)
            else:
                row[tag] = text

        return row

    def _parse_element_bs(self, element, include_attributes: bool) -> Dict:
        """Parse BeautifulSoup element to dictionary"""
        row = {}

        # Add attributes
        if include_attributes:
            for key, value in element.attrs.items():
                row[f"@{key}"] = value

        # Add child elements
        for child in element.children:
            if child.name is None:
                continue

            tag = child.name
            text = child.get_text(strip=True)

            # Handle nested elements
            if len(list(child.children)) > 1:
                row[tag] = self._parse_element_bs(child, include_attributes)
            else:
                row[tag] = text

        return row

    def _detect_record_tag(self) -> Optional[str]:
        """Detect likely record tag in XML"""
        if self._root is None:
            return None

        # Simple heuristic: find the most common child tag
        tag_counts = {}
        for child in self._root:
            tag = child.tag
            if '}' in tag:
                tag = tag.split('}')[1]
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if tag_counts:
            return max(tag_counts, key=tag_counts.get)

        return None

    def _detect_record_tag_bs(self, soup) -> Optional[str]:
        """Detect likely record tag in BeautifulSoup"""
        # Find most common tag
        tags = [tag.name for tag in soup.find_all()]
        if not tags:
            return None

        tag_counts = {}
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return max(tag_counts, key=tag_counts.get)

    def get_xml_structure(self) -> Dict[str, Any]:
        """
        Get structure information about the XML

        Returns:
            Dictionary describing XML structure
        """
        if self._root is None:
            raise DataImportError("No XML data has been imported")

        def analyze_element(element: ET.Element, path: str = "") -> Dict:
            tag = element.tag
            if '}' in tag:
                tag = tag.split('}')[1]

            current_path = f"{path}/{tag}" if path else tag

            children = list(element)

            return {
                "tag": tag,
                "path": current_path,
                "attributes": list(element.attrib.keys()),
                "has_text": bool(element.text and element.text.strip()),
                "children_count": len(children),
                "children": [
                    analyze_element(child, current_path)
                    for child in children[:5]  # Limit to first 5 for brevity
                ] if children else []
            }

        return analyze_element(self._root)
