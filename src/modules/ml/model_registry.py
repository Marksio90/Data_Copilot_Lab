"""
Data Copilot Lab - Model Registry
Save, load, version, and manage trained ML models
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import joblib

from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Model Registry for managing trained models

    Features:
    - Save/load models with metadata
    - Model versioning
    - Model comparison and selection
    - Metadata tracking (metrics, hyperparameters, training info)
    - Model search and filtering
    """

    def __init__(self, registry_path: Union[str, Path] = "./models"):
        """
        Initialize Model Registry

        Args:
            registry_path: Path to store models
        """
        self.logger = logger
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.registry_path / "registry_metadata.json"
        self.metadata = self._load_metadata()

    def save_model(
        self,
        model: Any,
        name: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        task_type: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model to the registry

        Args:
            model: Trained model object
            name: Model name
            version: Version string (auto-generated if None)
            metrics: Performance metrics
            hyperparameters: Model hyperparameters
            feature_names: List of feature names
            model_type: Type of model (e.g., 'random_forest', 'xgboost')
            task_type: 'classification', 'regression', or 'clustering'
            description: Model description
            tags: Tags for categorization
            additional_info: Any additional information

        Returns:
            Model ID
        """
        # Generate version if not provided
        if version is None:
            version = self._generate_version(name)

        # Create model ID
        model_id = f"{name}_v{version}"

        self.logger.info(f"Saving model: {model_id}")

        # Create model directory
        model_dir = self.registry_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model file
        model_path = model_dir / "model.pkl"
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            self.logger.error(f"Failed to save model with joblib: {e}")
            # Fallback to pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Prepare metadata
        metadata = {
            "model_id": model_id,
            "name": name,
            "version": version,
            "model_type": model_type,
            "task_type": task_type,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "feature_names": feature_names or [],
            "model_path": str(model_path.relative_to(self.registry_path)),
            "additional_info": additional_info or {}
        }

        # Save model metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update registry metadata
        self.metadata[model_id] = metadata
        self._save_metadata()

        self.logger.info(f"Model saved successfully: {model_id}")

        return model_id

    def load_model(self, model_id: Optional[str] = None, name: Optional[str] = None, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model from the registry

        Args:
            model_id: Full model ID (name_vversion)
            name: Model name (if model_id not provided)
            version: Model version (latest if not provided)

        Returns:
            Dictionary with 'model' and 'metadata'
        """
        # Resolve model_id
        if model_id is None:
            if name is None:
                raise InvalidParameterError("Either model_id or name must be provided")

            if version is None:
                # Get latest version
                version = self._get_latest_version(name)
                if version is None:
                    raise InvalidParameterError(f"No models found for name: {name}")

            model_id = f"{name}_v{version}"

        self.logger.info(f"Loading model: {model_id}")

        # Check if model exists
        if model_id not in self.metadata:
            raise InvalidParameterError(f"Model not found: {model_id}")

        metadata = self.metadata[model_id]
        model_path = self.registry_path / metadata['model_path']

        # Load model
        try:
            model = joblib.load(model_path)
        except Exception as e:
            self.logger.warning(f"Failed to load with joblib: {e}. Trying pickle...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        self.logger.info(f"Model loaded successfully: {model_id}")

        return {
            "model": model,
            "metadata": metadata
        }

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry

        Args:
            model_id: Model ID to delete

        Returns:
            True if deleted successfully
        """
        if model_id not in self.metadata:
            raise InvalidParameterError(f"Model not found: {model_id}")

        self.logger.info(f"Deleting model: {model_id}")

        metadata = self.metadata[model_id]
        model_path = self.registry_path / metadata['model_path']

        # Delete model file and directory
        if model_path.exists():
            model_path.unlink()

        model_dir = model_path.parent
        if model_dir.exists():
            # Delete metadata.json
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()

            # Remove directory if empty
            try:
                model_dir.rmdir()
            except:
                pass  # Directory not empty

        # Remove from metadata
        del self.metadata[model_id]
        self._save_metadata()

        self.logger.info(f"Model deleted: {model_id}")

        return True

    def list_models(
        self,
        name: Optional[str] = None,
        task_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = 'created_at',
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List models in the registry

        Args:
            name: Filter by model name
            task_type: Filter by task type
            tags: Filter by tags (models with any of these tags)
            sort_by: Sort by field ('created_at', 'name', or any metric)
            ascending: Sort order

        Returns:
            List of model metadata
        """
        models = list(self.metadata.values())

        # Apply filters
        if name:
            models = [m for m in models if m['name'] == name]

        if task_type:
            models = [m for m in models if m.get('task_type') == task_type]

        if tags:
            models = [m for m in models if any(tag in m.get('tags', []) for tag in tags)]

        # Sort
        if sort_by == 'created_at':
            models.sort(key=lambda x: x.get('created_at', ''), reverse=not ascending)
        elif sort_by == 'name':
            models.sort(key=lambda x: x.get('name', ''), reverse=not ascending)
        else:
            # Sort by metric
            models.sort(
                key=lambda x: x.get('metrics', {}).get(sort_by, 0),
                reverse=not ascending
            )

        return models

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get model metadata

        Args:
            model_id: Model ID

        Returns:
            Model metadata
        """
        if model_id not in self.metadata:
            raise InvalidParameterError(f"Model not found: {model_id}")

        return self.metadata[model_id]

    def compare_models(
        self,
        model_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            model_ids: List of model IDs to compare
            metrics: Specific metrics to compare (None = all metrics)

        Returns:
            DataFrame with comparison
        """
        comparison_data = []

        for model_id in model_ids:
            if model_id not in self.metadata:
                self.logger.warning(f"Model not found: {model_id}")
                continue

            metadata = self.metadata[model_id]

            row = {
                "model_id": model_id,
                "name": metadata['name'],
                "version": metadata['version'],
                "model_type": metadata.get('model_type'),
                "created_at": metadata.get('created_at')
            }

            # Add metrics
            model_metrics = metadata.get('metrics', {})
            if metrics:
                for metric in metrics:
                    row[metric] = model_metrics.get(metric)
            else:
                row.update(model_metrics)

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_best_model(
        self,
        name: Optional[str] = None,
        task_type: Optional[str] = None,
        metric: str = 'accuracy',
        higher_is_better: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get best model based on a metric

        Args:
            name: Filter by model name
            task_type: Filter by task type
            metric: Metric to optimize
            higher_is_better: True if higher metric is better

        Returns:
            Best model's metadata
        """
        models = self.list_models(name=name, task_type=task_type)

        if not models:
            return None

        # Filter models that have the metric
        models_with_metric = [m for m in models if metric in m.get('metrics', {})]

        if not models_with_metric:
            self.logger.warning(f"No models found with metric: {metric}")
            return None

        # Find best
        best_model = max(
            models_with_metric,
            key=lambda x: x['metrics'][metric] if higher_is_better else -x['metrics'][metric]
        )

        return best_model

    def update_model_metadata(
        self,
        model_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update model metadata

        Args:
            model_id: Model ID
            updates: Dictionary of updates

        Returns:
            True if updated successfully
        """
        if model_id not in self.metadata:
            raise InvalidParameterError(f"Model not found: {model_id}")

        # Update metadata
        self.metadata[model_id].update(updates)
        self.metadata[model_id]['updated_at'] = datetime.now().isoformat()

        # Save metadata file
        model_path = self.registry_path / self.metadata[model_id]['model_path']
        metadata_path = model_path.parent / "metadata.json"

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata[model_id], f, indent=2)

        self._save_metadata()

        self.logger.info(f"Metadata updated for: {model_id}")

        return True

    def export_model(
        self,
        model_id: str,
        export_path: Union[str, Path],
        format: str = 'joblib'
    ) -> str:
        """
        Export model to a specific path

        Args:
            model_id: Model ID
            export_path: Export destination
            format: Export format ('joblib', 'pickle')

        Returns:
            Export path
        """
        if model_id not in self.metadata:
            raise InvalidParameterError(f"Model not found: {model_id}")

        # Load model
        model_data = self.load_model(model_id)
        model = model_data['model']

        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'joblib':
            joblib.dump(model, export_path)
        elif format == 'pickle':
            with open(export_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise InvalidParameterError(f"Unknown format: {format}")

        # Also export metadata
        metadata_path = export_path.parent / f"{export_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_data['metadata'], f, indent=2)

        self.logger.info(f"Model exported to: {export_path}")

        return str(export_path)

    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """
        Search models by name, description, or tags

        Args:
            query: Search query

        Returns:
            List of matching models
        """
        query_lower = query.lower()
        results = []

        for model_id, metadata in self.metadata.items():
            # Search in name
            if query_lower in metadata.get('name', '').lower():
                results.append(metadata)
                continue

            # Search in description
            if query_lower in metadata.get('description', '').lower():
                results.append(metadata)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in metadata.get('tags', [])):
                results.append(metadata)
                continue

        return results

    def get_model_lineage(self, model_id: str) -> List[str]:
        """
        Get version lineage for a model

        Args:
            model_id: Model ID

        Returns:
            List of version IDs (sorted by creation time)
        """
        if model_id not in self.metadata:
            raise InvalidParameterError(f"Model not found: {model_id}")

        name = self.metadata[model_id]['name']

        # Get all versions of this model
        versions = [
            m for m in self.metadata.values()
            if m['name'] == name
        ]

        # Sort by creation time
        versions.sort(key=lambda x: x.get('created_at', ''))

        return [v['model_id'] for v in versions]

    def _generate_version(self, name: str) -> str:
        """Generate next version number for a model name"""
        existing_versions = [
            m['version'] for m in self.metadata.values()
            if m['name'] == name
        ]

        if not existing_versions:
            return "1.0"

        # Find highest version
        try:
            version_numbers = [
                tuple(map(int, v.split('.'))) for v in existing_versions if '.' in v
            ]
            if version_numbers:
                max_version = max(version_numbers)
                return f"{max_version[0]}.{max_version[1] + 1}"
        except:
            pass

        # Fallback: use timestamp
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def _get_latest_version(self, name: str) -> Optional[str]:
        """Get latest version for a model name"""
        versions = [
            m for m in self.metadata.values()
            if m['name'] == name
        ]

        if not versions:
            return None

        # Sort by creation time
        versions.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return versions[0]['version']

    def _load_metadata(self) -> Dict[str, Any]:
        """Load registry metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save registry metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            "total_models": len(self.metadata),
            "unique_names": len(set(m['name'] for m in self.metadata.values())),
            "task_types": {},
            "model_types": {},
            "recent_models": []
        }

        # Count by task type
        for metadata in self.metadata.values():
            task_type = metadata.get('task_type', 'unknown')
            stats['task_types'][task_type] = stats['task_types'].get(task_type, 0) + 1

            model_type = metadata.get('model_type', 'unknown')
            stats['model_types'][model_type] = stats['model_types'].get(model_type, 0) + 1

        # Get 5 most recent models
        recent = sorted(
            self.metadata.values(),
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )[:5]

        stats['recent_models'] = [
            {
                "model_id": m['model_id'],
                "name": m['name'],
                "created_at": m.get('created_at')
            }
            for m in recent
        ]

        return stats
