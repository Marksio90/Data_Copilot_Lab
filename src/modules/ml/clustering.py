"""
Data Copilot Lab - Clustering Models
Unsupervised learning and clustering algorithms
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    MeanShift,
    SpectralClustering,
    Birch,
    OPTICS,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

from src.core.exceptions import InvalidParameterError, ModelTrainingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClusteringMethod(str, Enum):
    """Available clustering methods"""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    MEAN_SHIFT = "mean_shift"
    SPECTRAL = "spectral"
    BIRCH = "birch"
    OPTICS = "optics"


class ClusteringTrainer:
    """
    Perform clustering analysis

    Supports 8+ clustering algorithms with automatic
    optimal cluster number detection and comprehensive evaluation
    """

    def __init__(self, random_state: int = 42):
        self.logger = logger
        self.random_state = random_state
        self.model = None
        self.method = None
        self.feature_names = None
        self.labels_ = None
        self.n_clusters_ = None
        self.cluster_centers_ = None
        self.clustering_info = {}

    def fit(
        self,
        data: pd.DataFrame,
        method: Union[str, ClusteringMethod] = ClusteringMethod.KMEANS,
        n_clusters: Optional[int] = None,
        features: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        auto_determine_clusters: bool = False,
        max_clusters: int = 10
    ) -> Dict[str, Any]:
        """
        Perform clustering

        Args:
            data: DataFrame
            method: Clustering method
            n_clusters: Number of clusters (None = auto-determine or method default)
            features: Feature columns (None = all numeric)
            hyperparameters: Method hyperparameters
            auto_determine_clusters: Automatically find optimal number of clusters
            max_clusters: Maximum clusters to try for auto-determination

        Returns:
            Clustering results and metrics
        """
        # Convert to enum
        if isinstance(method, str):
            method = ClusteringMethod(method)

        self.method = method

        self.logger.info(f"Performing {method.value} clustering")

        # Prepare data
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_names = features
        X = data[features]
        X = self._prepare_features(X)

        # Auto-determine optimal clusters
        if auto_determine_clusters and n_clusters is None:
            n_clusters = self._find_optimal_clusters(X, method, max_clusters)
            self.logger.info(f"Optimal number of clusters: {n_clusters}")

        # Create and fit model
        self.model = self._create_model(method, n_clusters, hyperparameters)

        try:
            if hasattr(self.model, 'fit_predict'):
                self.labels_ = self.model.fit_predict(X)
            else:
                self.model.fit(X)
                self.labels_ = self.model.labels_
        except Exception as e:
            raise ModelTrainingError(f"Clustering failed: {str(e)}")

        # Get cluster information
        self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))  # Exclude noise (-1)

        # Get cluster centers if available
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers_ = self.model.cluster_centers_
        elif method == ClusteringMethod.HIERARCHICAL:
            # Calculate centroids manually
            self.cluster_centers_ = self._calculate_centroids(X, self.labels_)

        # Evaluate clustering
        metrics = self._evaluate(X, self.labels_)

        # Analyze clusters
        cluster_analysis = self._analyze_clusters(data, self.labels_, features)

        # Store clustering info
        self.clustering_info = {
            "method": method.value,
            "n_features": len(features),
            "n_samples": len(X),
            "n_clusters": self.n_clusters_,
            "metrics": metrics,
            "cluster_analysis": cluster_analysis,
            "hyperparameters": hyperparameters or {}
        }

        self.logger.info(f"Clustering complete. Found {self.n_clusters_} clusters. "
                        f"Silhouette score: {metrics.get('silhouette', 'N/A'):.4f}")

        return self.clustering_info

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Assign clusters to new data

        Args:
            data: DataFrame with features

        Returns:
            Cluster labels
        """
        if self.model is None:
            raise ModelTrainingError("Model not fitted yet")

        X = data[self.feature_names]
        X = self._prepare_features(X)

        # Predict clusters
        if hasattr(self.model, 'predict'):
            labels = self.model.predict(X)
        else:
            # For methods without predict (like DBSCAN), assign to nearest center
            if self.cluster_centers_ is not None:
                distances = cdist(X, self.cluster_centers_)
                labels = np.argmin(distances, axis=1)
            else:
                raise InvalidParameterError(f"{self.method.value} does not support prediction")

        return labels

    def _create_model(
        self,
        method: ClusteringMethod,
        n_clusters: Optional[int],
        hyperparameters: Optional[Dict[str, Any]]
    ):
        """Create clustering model"""
        params = hyperparameters or {}

        if method == ClusteringMethod.KMEANS:
            n_clusters = n_clusters or 3
            return KMeans(n_clusters=n_clusters, random_state=self.random_state, **params)

        elif method == ClusteringMethod.DBSCAN:
            return DBSCAN(**params)

        elif method == ClusteringMethod.HIERARCHICAL:
            n_clusters = n_clusters or 3
            return AgglomerativeClustering(n_clusters=n_clusters, **params)

        elif method == ClusteringMethod.GAUSSIAN_MIXTURE:
            n_clusters = n_clusters or 3
            return GaussianMixture(n_components=n_clusters, random_state=self.random_state, **params)

        elif method == ClusteringMethod.MEAN_SHIFT:
            return MeanShift(**params)

        elif method == ClusteringMethod.SPECTRAL:
            n_clusters = n_clusters or 3
            return SpectralClustering(n_clusters=n_clusters, random_state=self.random_state, **params)

        elif method == ClusteringMethod.BIRCH:
            n_clusters = n_clusters or 3
            return Birch(n_clusters=n_clusters, **params)

        elif method == ClusteringMethod.OPTICS:
            return OPTICS(**params)

        else:
            raise InvalidParameterError(f"Unknown method: {method}")

    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare features for clustering"""
        X = X.copy()

        # Handle categorical features
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = pd.Categorical(X[col]).codes

        # Fill missing values
        X = X.fillna(X.mean())

        # Standardize (important for distance-based methods)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled

    def _evaluate(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate clustering quality"""
        metrics = {}

        # Filter out noise points (label -1 for DBSCAN)
        valid_mask = labels >= 0
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]

        n_clusters = len(np.unique(labels_valid))

        # Need at least 2 clusters for most metrics
        if n_clusters < 2:
            self.logger.warning("Less than 2 clusters found. Metrics may be limited.")
            return {"n_clusters": n_clusters, "message": "Insufficient clusters for evaluation"}

        try:
            # Silhouette score (-1 to 1, higher is better)
            silhouette = silhouette_score(X_valid, labels_valid)
            metrics["silhouette"] = float(silhouette)
        except Exception as e:
            self.logger.debug(f"Could not calculate silhouette score: {e}")

        try:
            # Calinski-Harabasz score (higher is better)
            calinski = calinski_harabasz_score(X_valid, labels_valid)
            metrics["calinski_harabasz"] = float(calinski)
        except Exception as e:
            self.logger.debug(f"Could not calculate Calinski-Harabasz score: {e}")

        try:
            # Davies-Bouldin score (lower is better)
            davies = davies_bouldin_score(X_valid, labels_valid)
            metrics["davies_bouldin"] = float(davies)
        except Exception as e:
            self.logger.debug(f"Could not calculate Davies-Bouldin score: {e}")

        # Inertia (for K-Means)
        if hasattr(self.model, 'inertia_'):
            metrics["inertia"] = float(self.model.inertia_)

        # Noise ratio (for DBSCAN/OPTICS)
        n_noise = np.sum(labels == -1)
        if n_noise > 0:
            metrics["noise_ratio"] = float(n_noise / len(labels))
            metrics["n_noise_points"] = int(n_noise)

        return metrics

    def _analyze_clusters(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        features: List[str]
    ) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        # Add cluster labels to data
        data_with_clusters = data[features].copy()
        data_with_clusters['cluster'] = labels

        analysis = {}

        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Noise
                continue

            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]

            cluster_info = {
                "size": len(cluster_data),
                "percentage": float(len(cluster_data) / len(data) * 100),
                "feature_means": cluster_data[features].mean().to_dict(),
                "feature_stds": cluster_data[features].std().to_dict()
            }

            analysis[f"cluster_{cluster_id}"] = cluster_info

        return analysis

    def _find_optimal_clusters(
        self,
        X: np.ndarray,
        method: ClusteringMethod,
        max_clusters: int = 10
    ) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score

        Args:
            X: Feature matrix
            method: Clustering method
            max_clusters: Maximum clusters to try

        Returns:
            Optimal number of clusters
        """
        self.logger.info(f"Finding optimal number of clusters (2 to {max_clusters})")

        if method not in [ClusteringMethod.KMEANS, ClusteringMethod.HIERARCHICAL, ClusteringMethod.GAUSSIAN_MIXTURE]:
            self.logger.warning(f"{method.value} does not support cluster number specification. Using default.")
            return 3

        scores = []
        inertias = []

        for k in range(2, max_clusters + 1):
            if method == ClusteringMethod.KMEANS:
                model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = model.fit_predict(X)
                if hasattr(model, 'inertia_'):
                    inertias.append(model.inertia_)
            elif method == ClusteringMethod.HIERARCHICAL:
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X)
            elif method == ClusteringMethod.GAUSSIAN_MIXTURE:
                model = GaussianMixture(n_components=k, random_state=self.random_state)
                labels = model.fit_predict(X)
            else:
                continue

            # Calculate silhouette score
            try:
                score = silhouette_score(X, labels)
                scores.append(score)
            except:
                scores.append(0)

        # Find best by silhouette score
        if scores:
            optimal_k = np.argmax(scores) + 2  # +2 because we started from 2
            self.logger.info(f"Silhouette scores: {[f'{s:.3f}' for s in scores]}")

            # If using K-Means, also check elbow method
            if inertias and len(inertias) > 2:
                elbow_k = self._find_elbow(inertias) + 2
                self.logger.info(f"Elbow method suggests: {elbow_k} clusters")

                # If elbow and silhouette disagree, prefer smaller number
                if abs(optimal_k - elbow_k) > 2:
                    optimal_k = min(optimal_k, elbow_k)

            return optimal_k

        return 3  # Default fallback

    def _find_elbow(self, inertias: List[float]) -> int:
        """Find elbow point in inertia curve"""
        # Simple elbow detection using angle method
        angles = []
        for i in range(1, len(inertias) - 1):
            # Calculate angle between three points
            p1 = np.array([i - 1, inertias[i - 1]])
            p2 = np.array([i, inertias[i]])
            p3 = np.array([i + 1, inertias[i + 1]])

            v1 = p1 - p2
            v2 = p3 - p2

            # Cosine of angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)

        # Find maximum angle (sharpest turn)
        if angles:
            elbow_idx = np.argmax(angles) + 1  # +1 because we started from index 1
            return elbow_idx

        return len(inertias) // 2

    def _calculate_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centroids manually"""
        centroids = []
        for cluster_id in np.unique(labels):
            if cluster_id >= 0:  # Exclude noise
                cluster_points = X[labels == cluster_id]
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid)

        return np.array(centroids)

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers"""
        return self.cluster_centers_

    def get_labels(self) -> Optional[np.ndarray]:
        """Get cluster labels"""
        return self.labels_

    def get_clustering_info(self) -> Dict[str, Any]:
        """Get clustering information and metrics"""
        return self.clustering_info

    def visualize_dendrogram(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate dendrogram data for hierarchical clustering visualization

        Args:
            data: DataFrame
            features: Features to use (None = all numeric)

        Returns:
            Dictionary with linkage matrix and dendrogram data
        """
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        X = data[features]
        X = self._prepare_features(X)

        # Perform hierarchical clustering
        linkage_matrix = linkage(X, method='ward')

        return {
            "linkage_matrix": linkage_matrix.tolist(),
            "n_samples": len(X),
            "method": "ward",
            "message": "Use this linkage matrix with scipy.cluster.hierarchy.dendrogram for visualization"
        }

    def compare_methods(
        self,
        data: pd.DataFrame,
        methods: Optional[List[ClusteringMethod]] = None,
        n_clusters: int = 3,
        features: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple clustering methods

        Args:
            data: DataFrame
            methods: Methods to compare (None = all applicable methods)
            n_clusters: Number of clusters
            features: Features to use

        Returns:
            Comparison results
        """
        if methods is None:
            methods = [
                ClusteringMethod.KMEANS,
                ClusteringMethod.HIERARCHICAL,
                ClusteringMethod.GAUSSIAN_MIXTURE,
                ClusteringMethod.DBSCAN
            ]

        self.logger.info(f"Comparing {len(methods)} clustering methods")

        results = {}

        for method in methods:
            try:
                # Temporarily store original model
                original_model = self.model
                original_labels = self.labels_

                # Fit with this method
                method_results = self.fit(
                    data,
                    method=method,
                    n_clusters=n_clusters if method != ClusteringMethod.DBSCAN else None,
                    features=features
                )

                results[method.value] = {
                    "n_clusters": method_results["n_clusters"],
                    "metrics": method_results["metrics"]
                }

                # Restore original
                self.model = original_model
                self.labels_ = original_labels

            except Exception as e:
                self.logger.warning(f"Failed to fit {method.value}: {e}")
                results[method.value] = {"error": str(e)}

        return results
