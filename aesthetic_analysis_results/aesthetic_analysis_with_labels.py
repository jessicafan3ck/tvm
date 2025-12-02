import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Data processing and analysis
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Feature correlation and statistical analysis
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)


class AestheticFeatureExtractor:
    """Extract and organize aesthetic features from per-file JSON results with labels."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.raw_features = []
        self.metadata = []
        self.labels = []  # Store modifier/subject labels
        self.feature_df = None
        
    def load_all_files(self) -> pd.DataFrame:
        """Load all JSON files from directory, extracting labels from filenames."""
        json_files = sorted(list(self.data_dir.glob('*.json')))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                self.raw_features.append(data['features'])
                self.metadata.append(data['meta'])
                
                # Extract label from filename (format: number_[label].json)
                filename = json_file.stem  # Remove .json extension
                parts = filename.split('_', 1)  # Split on first underscore
                if len(parts) == 2:
                    label = parts[1]
                else:
                    label = filename
                self.labels.append(label)
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return self._flatten_to_dataframe()
    
    def _flatten_to_dataframe(self) -> pd.DataFrame:
        """Convert nested feature dictionaries into flat dataframe."""
        flattened = []
        
        for meta, features, label in zip(self.metadata, self.raw_features, self.labels):
            row = {
                '_filename': meta['_filename'],
                'label': label  # Add label column
            }
            
            # Flatten nested dictionaries
            for category, subcategories in features.items():
                if isinstance(subcategories, dict):
                    for subcat, values in subcategories.items():
                        if isinstance(values, dict):
                            for key, val in values.items():
                                if isinstance(val, (int, float)):
                                    row[f"{category}_{subcat}_{key}"] = val
                        elif isinstance(values, (int, float)):
                            row[f"{category}_{subcat}"] = values
                elif isinstance(subcategories, (int, float)):
                    row[f"{category}"] = subcategories
            
            flattened.append(row)
        
        self.feature_df = pd.DataFrame(flattened)
        print(f"Extracted {len(self.feature_df)} samples with {len(self.feature_df.columns)} features")
        print(f"Unique labels: {len(self.feature_df['label'].unique())}")
        print(f"Labels: {sorted(self.feature_df['label'].unique())}")
        return self.feature_df


class ObjectiveFeatureAnalyzer:
    """Analyze objective visual features (F_obj: geometric, spatial, contrast, etc.)"""
    
    # Define feature categories based on your data structure
    FEATURE_CATEGORIES = {
        'geometric_structural': [
            'leading_lines_strength', 'symmetry_vertical', 'symmetry_horizontal',
            'symmetry_radial', 'balance_static_balance', 'balance_dynamic_balance',
            'layering_foreground_ratio', 'layering_midground_ratio', 'layering_background_ratio',
            'layering_depth_complexity'
        ],
        'spatial_layout': [
            'negative_space_ratio', 'subject_isolation', 'depth_of_field_blur_gradient_strength',
            'depth_of_field_sharpness_variance', 'perspective_has_vanishing_point',
            'perspective_perspective_strength'
        ],
        'contrast_color': [
            'high_contrast_ratio', 'low_contrast_ratio', 'mean_contrast',
            'highlight_ratio', 'shadow_ratio', 'midtone_ratio',
            'warm_color_ratio', 'cool_color_ratio', 'complementary_color_score'
        ],
        'motion_dynamics': [
            's_curve_count', 'motion_blur_score', 'dominant_direction_deg',
            'direction_strength'
        ],
        'pattern_rhythm': [
            'repetition_score', 'rhythm_score', 'pattern_breaks'
        ],
        'visual_mass_focus': [
            'visual_mass_concentration', 'mean_visual_weight', 'face_count',
            'has_directional_cues'
        ],
        'framing_breathing': [
            'breathing_space_top', 'breathing_space_bottom', 'breathing_space_left',
            'breathing_space_right', 'breathing_space_mean_margin'
        ]
    }
    
    def __init__(self, feature_df: pd.DataFrame):
        self.feature_df = feature_df.copy()
        self.objective_features = None
        self.scaler = RobustScaler()  # Robust to outliers in aesthetic data
        self.scaled_features = None
        
    def extract_objective_features(self) -> np.ndarray:
        """Extract F_obj: objective visual features."""
        candidate_cols = self._get_all_feature_names()
        available_cols = [col for col in candidate_cols if col in self.feature_df.columns]

        if not available_cols:
            print("\n[ERROR] No objective feature columns from FEATURE_CATEGORIES "
                  "were found in feature_df.")
            print(f"feature_df has {self.feature_df.shape[1]} columns. "
                  "First 40 columns:")
            print(list(self.feature_df.columns)[:40])
            raise ValueError(
                "No objective features found. Check that your decomposition step "
                "produced columns matching FEATURE_CATEGORIES."
            )
        
        self.objective_features = self.feature_df[available_cols].fillna(0)
        self.scaled_features = self.scaler.fit_transform(self.objective_features)
        
        print(f"Extracted {self.objective_features.shape[1]} objective features")
        return self.scaled_features
    
    def _get_all_feature_names(self) -> List[str]:
        """Get all feature names from categories, with category prefixes."""
        all_features = []
        for category_name, category_features in self.FEATURE_CATEGORIES.items():
            for f in category_features:
                full_name = f"{category_name}_{f}"
                all_features.append(full_name)
        return all_features
    
    def feature_importance_by_category(self) -> Dict[str, float]:
        """Calculate variance contribution of each feature category."""
        pca = PCA()
        pca.fit(self.scaled_features)
        
        variance_by_category = {}
        for category, features in self.FEATURE_CATEGORIES.items():
            available = [
                f"{category}_{f}" for f in features
                if f"{category}_{f}" in self.objective_features.columns
            ]
            if available:
                feature_indices = [
                    list(self.objective_features.columns).index(f) 
                    for f in available
                ]
                variance_by_category[category] = np.mean(
                    pca.explained_variance_ratio_[feature_indices]
                )
        
        return variance_by_category
    
    def correlation_analysis(self) -> pd.DataFrame:
        """Analyze correlations between feature categories."""
        category_means = {}
        for category, features in self.FEATURE_CATEGORIES.items():
            available = [
                f"{category}_{f}" for f in features
                if f"{category}_{f}" in self.objective_features.columns
            ]
            if available:
                category_means[category] = self.objective_features[available].mean(axis=1)
        
        category_df = pd.DataFrame(category_means)
        return category_df.corr()


class DimensionalityReducer:
    """Apply multiple dimensionality reduction techniques."""
    
    def __init__(self, scaled_features: np.ndarray, n_neighbors: int = 15):
        self.scaled_features = scaled_features
        self.n_neighbors = n_neighbors
        self.reductions = {}
        
    def apply_pca(self, n_components: int = 50) -> Tuple[np.ndarray, float]:
        """PCA for interpretability."""
        max_components = min(self.scaled_features.shape[0], self.scaled_features.shape[1])
        if n_components > max_components:
            print(
                f"Requested {n_components} PCA components but only "
                f"{max_components} features are available; using {max_components}."
            )
            n_components = max_components

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(self.scaled_features)
        explained_variance = pca.explained_variance_ratio_.sum()
        self.reductions['pca'] = {'model': pca, 'data': reduced, 'variance': explained_variance}
        print(f"PCA: {n_components} components explain {explained_variance:.2%} variance")
        return reduced, explained_variance
    
    def apply_umap(self, n_components: int = 20, metric: str = 'euclidean') -> np.ndarray:
        """UMAP for preserving global structure."""
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=self.n_neighbors,
            metric=metric,
            random_state=42,
            verbose=False
        )
        reduced = umap_reducer.fit_transform(self.scaled_features)
        self.reductions['umap'] = {'model': umap_reducer, 'data': reduced}
        print(f"UMAP: Reduced to {n_components} dimensions")
        return reduced
    
    def apply_tsne(self, n_components: int = 2, perplexity: int = 30) -> np.ndarray:
        """t-SNE for visualization (only use for final visualization, not analysis)."""
        try:
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=42,
                max_iter=1000
            )
        except TypeError:
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=42,
                n_iter=1000
            )

        reduced = tsne.fit_transform(self.scaled_features)
        self.reductions['tsne'] = {'model': tsne, 'data': reduced}
        print(f"t-SNE: Reduced to {n_components} dimensions")
        return reduced
    
    def compare_reductions(self) -> pd.DataFrame:
        """Compare quality of different reductions using reconstruction error."""
        results = {}
        for name, reduction_dict in self.reductions.items():
            if name == 'pca':
                reconstructed = reduction_dict['model'].inverse_transform(reduction_dict['data'])
                error = np.mean(
                    np.sqrt(np.sum((self.scaled_features - reconstructed) ** 2, axis=1))
                )
                results[name] = {'reconstruction_error': error}
            else:
                distance_original = pdist(self.scaled_features, metric='euclidean')
                distance_reduced = pdist(reduction_dict['data'], metric='euclidean')
                correlation = spearmanr(distance_original, distance_reduced)[0]
                results[name] = {'distance_correlation': correlation}
        
        return pd.DataFrame(results).T


class AestheticClusterer:
    """Cluster images based on aesthetic features with label alignment evaluation."""
    
    def __init__(self, features: np.ndarray, filenames: List[str] = None, 
                 labels: List[str] = None):
        self.features = features
        self.filenames = filenames
        self.labels = labels  # Ground truth aesthetic modifier labels
        self.clusters = {}
        self.optimal_k = None
        
    def find_optimal_clusters(self, k_range: range = range(2, 16)) -> Dict[str, List]:
        """Find optimal number of clusters using multiple metrics."""
        metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.features)
            
            metrics['silhouette'].append(silhouette_score(self.features, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(self.features, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(self.features, labels))
        
        optimal_silhouette = k_range[np.argmax(metrics['silhouette'])]
        optimal_db = k_range[np.argmin(metrics['davies_bouldin'])]
        optimal_ch = k_range[np.argmax(metrics['calinski_harabasz'])]
        
        self.optimal_k = optimal_silhouette
        
        return {
            'k_range': list(k_range),
            'silhouette_scores': metrics['silhouette'],
            'davies_bouldin_scores': metrics['davies_bouldin'],
            'calinski_harabasz_scores': metrics['calinski_harabasz'],
            'optimal_k_silhouette': optimal_silhouette,
            'optimal_k_davies_bouldin': optimal_db,
            'optimal_k_calinski_harabasz': optimal_ch
        }
    
    def kmeans_clustering(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """Apply K-means clustering."""
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 8
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.features)
        self.clusters['kmeans'] = {'model': kmeans, 'labels': labels, 'n_clusters': n_clusters}
        print(f"K-means: Clustered into {n_clusters} clusters")
        return labels
    
    def hierarchical_clustering(self, n_clusters: int = 8, linkage_method: str = 'ward') -> np.ndarray:
        """Apply hierarchical clustering."""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = hierarchical.fit_predict(self.features)
        self.clusters['hierarchical'] = {
            'model': hierarchical,
            'labels': labels,
            'n_clusters': n_clusters
        }
        print(f"Hierarchical ({linkage_method}): Clustered into {n_clusters} clusters")
        return labels
    
    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """Apply DBSCAN for density-based clustering."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.clusters['dbscan'] = {
            'model': dbscan,
            'labels': labels,
            'n_clusters': n_clusters
        }
        print(f"DBSCAN: Found {n_clusters} clusters (eps={eps}, min_samples={min_samples})")
        return labels
    
    def get_cluster_silhouettes(self, labels: np.ndarray) -> Dict:
        """Get silhouette scores for each cluster."""
        mask = labels >= 0
        if mask.sum() <= 1 or len(np.unique(labels[mask])) <= 1:
            return {
                'overall_silhouette': float('nan'),
                'per_sample_silhouettes': mask
            }
        silhouette_scores = silhouette_score(self.features[mask], labels[mask])
        return {
            'overall_silhouette': silhouette_scores,
            'per_sample_silhouettes': mask
        }
    
    def evaluate_label_alignment(self, predicted_labels: np.ndarray) -> Dict:
        """
        Evaluate how well predicted clusters align with ground truth aesthetic labels.
        This tests if the latent space naturally organizes by aesthetic modifier.
        """
        if self.labels is None:
            print("No ground truth labels available for alignment evaluation.")
            return {}
        
        # Convert string labels to numeric codes
        unique_labels = sorted(set(self.labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        true_labels = np.array([label_to_idx[l] for l in self.labels])
        
        # Calculate alignment metrics
        ami = adjusted_mutual_info_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        
        # Per-label purity: for each ground truth label, what's the most common cluster?
        label_purities = {}
        for true_label in unique_labels:
            mask = true_labels == label_to_idx[true_label]
            if mask.sum() > 0:
                cluster_assignment = predicted_labels[mask]
                most_common_cluster = np.argmax(np.bincount(cluster_assignment))
                purity = np.sum(cluster_assignment == most_common_cluster) / mask.sum()
                label_purities[true_label] = purity
        
        return {
            'adjusted_mutual_info': ami,
            'normalized_mutual_info': nmi,
            'adjusted_rand_index': ari,
            'label_purities': label_purities,
            'mean_purity': np.mean(list(label_purities.values()))
        }


class VibeLatentSpaceAnalyzer:
    """Analyze aesthetic latent space (Z_vibe) structure and aesthetic sentiment."""
    
    def __init__(self, reduced_features: np.ndarray, objective_features: np.ndarray,
                 filenames: List[str] = None, labels: List[str] = None):
        self.reduced_features = reduced_features  # Z_vibe approximation
        self.objective_features = objective_features  # F_obj
        self.filenames = filenames
        self.labels = labels  # Aesthetic modifier labels
        
    def analyze_modifier_separation(self, label_array: np.ndarray = None) -> Dict:
        """
        Analyze how well aesthetic modifiers separate in latent space.
        This tests if different modifiers occupy distinct regions in Z_vibe.
        """
        if label_array is None and self.labels is None:
            return {}
        
        if label_array is None:
            # Use provided labels
            unique_labels = sorted(set(self.labels))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            label_array = np.array([label_to_idx[l] for l in self.labels])
        
        unique_modifiers = np.unique(label_array)
        modifier_centroids = {}
        modifier_dispersions = {}
        
        for modifier in unique_modifiers:
            mask = label_array == modifier
            centroid = self.reduced_features[mask].mean(axis=0)
            modifier_centroids[modifier] = centroid
            
            distances_to_centroid = np.linalg.norm(
                self.reduced_features[mask] - centroid, axis=1
            )
            modifier_dispersions[modifier] = {
                'mean_distance': distances_to_centroid.mean(),
                'std_distance': distances_to_centroid.std(),
                'max_distance': distances_to_centroid.max(),
                'count': mask.sum()
            }
        
        # Inter-modifier distances
        centroid_distances = pairwise_distances(
            np.array(list(modifier_centroids.values())),
            metric='euclidean'
        )
        
        inter_distances = centroid_distances[
            np.triu_indices_from(centroid_distances, k=1)
        ]
        
        mean_intra_distance = np.mean(
            [d['mean_distance'] for d in modifier_dispersions.values()]
        )
        
        separation_index = inter_distances.mean() / mean_intra_distance if mean_intra_distance > 0 else 0
        
        return {
            'modifier_centroids': modifier_centroids,
            'modifier_dispersions': modifier_dispersions,
            'inter_modifier_distance_mean': float(inter_distances.mean()),
            'inter_modifier_distance_std': float(inter_distances.std()),
            'inter_modifier_distance_min': float(inter_distances.min()),
            'inter_modifier_distance_max': float(inter_distances.max()),
            'separation_index': float(separation_index),
            'n_modifiers': len(unique_modifiers)
        }


class AestheticVisualization:
    """Visualization tools for aesthetic latent space analysis."""
    
    @staticmethod
    def plot_dimensionality_reduction(reduced_features: np.ndarray, labels: np.ndarray = None,
                                      label_names: List[str] = None,
                                      title: str = "Dimensionality Reduction", 
                                      figsize: Tuple = (12, 8)):
        """Plot 2D or 3D dimensionality reduction with optional label coloring."""
        fig = plt.figure(figsize=figsize)
        
        if reduced_features.shape[1] >= 3:
            ax = fig.add_subplot(111, projection='3d')
            if labels is not None:
                scatter = ax.scatter(
                    reduced_features[:, 0], reduced_features[:, 1],
                    reduced_features[:, 2], c=labels, cmap='tab20', s=30
                )
                plt.colorbar(scatter, ax=ax, label='Cluster/Label')
            else:
                ax.scatter(
                    reduced_features[:, 0], reduced_features[:, 1],
                    reduced_features[:, 2], s=30, alpha=0.6
                )
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        else:
            ax = fig.add_subplot(111)
            if labels is not None:
                scatter = ax.scatter(
                    reduced_features[:, 0], reduced_features[:, 1],
                    c=labels, cmap='tab20', s=30, alpha=0.7
                )
                plt.colorbar(scatter, ax=ax, label='Cluster/Label')
            else:
                ax.scatter(reduced_features[:, 0], reduced_features[:, 1], s=30, alpha=0.6)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cluster_metrics(metrics_dict: Dict[str, List], k_range: List[int],
                             figsize: Tuple = (15, 5)):
        """Plot clustering evaluation metrics."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].plot(k_range, metrics_dict['silhouette_scores'], 'o-', linewidth=2)
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Silhouette Analysis')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(metrics_dict['optimal_k_silhouette'], color='r', linestyle='--', alpha=0.7)
        
        axes[1].plot(k_range, metrics_dict['davies_bouldin_scores'], 'o-', linewidth=2)
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Davies-Bouldin Index')
        axes[1].set_title('Davies-Bouldin Analysis (Lower is Better)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(metrics_dict['optimal_k_davies_bouldin'], color='r', linestyle='--', alpha=0.7)
        
        axes[2].plot(k_range, metrics_dict['calinski_harabasz_scores'], 'o-', linewidth=2)
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Calinski-Harabasz Index')
        axes[2].set_title('Calinski-Harabasz Analysis')
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(metrics_dict['optimal_k_calinski_harabasz'], color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_label_alignment(label_purities: Dict[str, float], figsize: Tuple = (12, 6)):
        """Visualize per-modifier cluster purity."""
        fig, ax = plt.subplots(figsize=figsize)
        
        modifiers = sorted(label_purities.keys())
        purities = [label_purities[m] for m in modifiers]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(modifiers)))
        bars = ax.barh(modifiers, purities, color=colors)
        ax.set_xlabel('Cluster Purity (Higher is Better)')
        ax.set_title('Per-Modifier Cluster Alignment with Ground Truth Labels')
        ax.set_xlim([0, 1])
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=9
            )
        
        plt.tight_layout()
        return fig


class AestheticAnalysisPipeline:
    """Complete pipeline orchestrating the analysis with label alignment."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.extractor = None
        self.objective_analyzer = None
        self.reducer = None
        self.clusterer = None
        self.vibe_analyzer = None
        self.results = {}
        
    def run_full_analysis(self, n_pca_components: int = 50, 
                          n_umap_components: int = 20,
                          n_clusters: int = None):
        """Execute complete analysis pipeline with label evaluation."""
        
        print("=" * 80)
        print("AESTHETIC LATENT SPACE ANALYSIS PIPELINE WITH LABEL ALIGNMENT")
        print("=" * 80)
        
        # 1. Feature Extraction
        print("\n[1/7] Extracting Features from Labeled JSONs...")
        self.extractor = AestheticFeatureExtractor(self.data_dir)
        feature_df = self.extractor.load_all_files()
        
        # Store labels for later evaluation
        true_labels = feature_df['label'].values
        
        # 2. Objective Feature Analysis
        print("\n[2/7] Analyzing Objective Features (F_obj)...")
        self.objective_analyzer = ObjectiveFeatureAnalyzer(feature_df)
        scaled_features = self.objective_analyzer.extract_objective_features()
        
        importance = self.objective_analyzer.feature_importance_by_category()
        self.results['feature_importance'] = importance
        print("Feature Importance by Category:")
        for cat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {imp:.4f}")
        
        corr_matrix = self.objective_analyzer.correlation_analysis()
        self.results['feature_correlations'] = corr_matrix
        print("\nTop Feature Category Correlations:")
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        print(f"  Mean: {corr_values.mean():.4f}, Max: {corr_values.max():.4f}")
        
        # 3. Dimensionality Reduction
        print("\n[3/7] Applying Dimensionality Reduction...")
        self.reducer = DimensionalityReducer(scaled_features, n_neighbors=15)
        
        pca_reduced, pca_variance = self.reducer.apply_pca(n_components=n_pca_components)
        umap_reduced = self.reducer.apply_umap(n_components=n_umap_components)
        tsne_reduced = self.reducer.apply_tsne(n_components=2)
        
        reduction_comparison = self.reducer.compare_reductions()
        self.results['reduction_comparison'] = reduction_comparison
        print("\nDimensionality Reduction Comparison:")
        print(reduction_comparison)
        
        # 4. Clustering
        print("\n[4/7] Clustering Analysis...")
        self.clusterer = AestheticClusterer(
            umap_reduced, 
            filenames=feature_df['_filename'].values if '_filename' in feature_df.columns else None,
            labels=true_labels
        )
        
        # Find optimal clusters
        optimal_metrics = self.clusterer.find_optimal_clusters(k_range=range(2, 30))
        self.results['optimal_clusters'] = optimal_metrics
        print(f"\nOptimal k (Silhouette): {optimal_metrics['optimal_k_silhouette']}")
        print(f"Optimal k (Davies-Bouldin): {optimal_metrics['optimal_k_davies_bouldin']}")
        print(f"Optimal k (Calinski-Harabasz): {optimal_metrics['optimal_k_calinski_harabasz']}")
        
        # Apply clustering with number of unique labels if not specified
        if n_clusters is None:
            n_unique_labels = len(set(true_labels))
            n_clusters = n_unique_labels
            print(f"\nUsing n_clusters={n_clusters} (matching {n_unique_labels} unique aesthetic modifiers)")
        
        kmeans_labels = self.clusterer.kmeans_clustering(n_clusters=n_clusters)
        hierarchical_labels = self.clusterer.hierarchical_clustering(
            n_clusters=n_clusters, linkage_method='ward'
        )
        dbscan_labels = self.clusterer.dbscan_clustering(eps=0.5, min_samples=5)
        
        self.results['kmeans_labels'] = kmeans_labels
        self.results['hierarchical_labels'] = hierarchical_labels
        self.results['dbscan_labels'] = dbscan_labels
        
        # 5. Label Alignment Evaluation
        print("\n[5/7] Evaluating Cluster Alignment with Ground Truth Labels...")
        kmeans_alignment = self.clusterer.evaluate_label_alignment(kmeans_labels)
        self.results['kmeans_alignment'] = kmeans_alignment
        
        print("\nK-Means Alignment with Aesthetic Modifiers:")
        print(f"  Adjusted Mutual Information: {kmeans_alignment.get('adjusted_mutual_info', 0):.4f}")
        print(f"  Normalized Mutual Information: {kmeans_alignment.get('normalized_mutual_info', 0):.4f}")
        print(f"  Adjusted Rand Index: {kmeans_alignment.get('adjusted_rand_index', 0):.4f}")
        print(f"  Mean Purity: {kmeans_alignment.get('mean_purity', 0):.4f}")
        
        print("\n  Per-Modifier Purity:")
        for modifier, purity in sorted(kmeans_alignment.get('label_purities', {}).items()):
            print(f"    {modifier}: {purity:.4f}")
        
        # 6. Latent Space Analysis (Z_vibe)
        print("\n[6/7] Analyzing Aesthetic Latent Space (Z_vibe)...")
        self.vibe_analyzer = VibeLatentSpaceAnalyzer(
            umap_reduced, scaled_features,
            filenames=feature_df['_filename'].values if '_filename' in feature_df.columns else None,
            labels=true_labels
        )
        
        # Analyze how well modifiers separate in latent space
        modifier_analysis = self.vibe_analyzer.analyze_modifier_separation()
        self.results['modifier_separation'] = {
            'inter_modifier_distance_mean': float(
                modifier_analysis['inter_modifier_distance_mean']
            ),
            'inter_modifier_distance_std': float(
                modifier_analysis['inter_modifier_distance_std']
            ),
            'separation_index': float(modifier_analysis['separation_index']),
            'n_modifiers': modifier_analysis['n_modifiers']
        }
        print(f"Inter-modifier distance (mean): "
              f"{modifier_analysis['inter_modifier_distance_mean']:.4f}")
        print(f"Inter-modifier distance (std): "
              f"{modifier_analysis['inter_modifier_distance_std']:.4f}")
        print(f"Separation index: {modifier_analysis['separation_index']:.4f}")
        print(f"Number of aesthetic modifiers: {modifier_analysis['n_modifiers']}")
        
        # 7. Summary Report
        print("\n[7/7] Generating Summary Report...")
        summary = self._generate_summary_report(
            feature_df, scaled_features, 
            pca_reduced, umap_reduced, tsne_reduced,
            kmeans_labels, kmeans_alignment
        )
        self.results['summary'] = summary
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE - KEY FINDINGS:")
        print("=" * 80)
        print(f"Modifier Separation Index: {self.results['modifier_separation']['separation_index']:.4f}")
        print(f"  (>1.0 indicates modifiers form well-separated clusters)")
        print(f"Cluster Purity: {kmeans_alignment.get('mean_purity', 0):.4f}")
        print(f"  (Measures how well predicted clusters align with ground truth modifiers)")
        print(f"NMI Score: {kmeans_alignment.get('normalized_mutual_info', 0):.4f}")
        print(f"  (0-1 scale, 1.0 is perfect alignment)")
        
        return self.results
    
    def _generate_summary_report(self, feature_df, scaled_features, pca_reduced, 
                                 umap_reduced, tsne_reduced, labels, alignment_metrics):
        """Generate comprehensive summary report."""
        silhouettes = self.clusterer.get_cluster_silhouettes(labels)
        return {
            'total_samples': len(feature_df),
            'total_features': scaled_features.shape[1],
            'pca_variance_explained': float(self.reducer.reductions['pca']['variance']),
            'reduced_dimensions_umap': umap_reduced.shape[1],
            'reduced_dimensions_pca': pca_reduced.shape[1],
            'n_clusters': len(np.unique(labels)),
            'silhouette_score': float(silhouettes['overall_silhouette']),
            'adjusted_mutual_info': float(alignment_metrics.get('adjusted_mutual_info', 0)),
            'normalized_mutual_info': float(alignment_metrics.get('normalized_mutual_info', 0)),
            'adjusted_rand_index': float(alignment_metrics.get('adjusted_rand_index', 0)),
            'mean_purity': float(alignment_metrics.get('mean_purity', 0))
        }
    
    def save_results(self, output_dir: str = './aesthetic_analysis_results'):
        """Save analysis results to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON
        summary_data = {
            k: v for k, v in self.results.items() 
            if k not in ['feature_correlations', 'reduction_comparison']
        }
        
        with open(output_path / 'analysis_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Save correlation matrix
        if 'feature_correlations' in self.results:
            self.results['feature_correlations'].to_csv(
                output_path / 'feature_correlations.csv'
            )
        
        # Save reduction comparison
        if 'reduction_comparison' in self.results:
            self.results['reduction_comparison'].to_csv(
                output_path / 'reduction_comparison.csv'
            )
        
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    # Initialize pipeline with path to labeled JSON files
    data_directory = (
        '/Users/jessicafan/Downloads/'
        'tvm board - Nov 22nd 2025 (5060 images)/'
        'TVM JSON'
    )
    
    pipeline = AestheticAnalysisPipeline(data_directory)
    results = pipeline.run_full_analysis(
        n_pca_components=50,
        n_umap_components=20,
        n_clusters=None  # Will auto-determine based on unique labels
    )
    
    # Save results
    pipeline.save_results('./aesthetic_analysis_results')
    
    # Access specific results
    print("\n\nKEY FINDINGS:")
    print(f"Modifier Separation Index: "
          f"{results['modifier_separation']['separation_index']:.4f}")
    print(f"Silhouette Score: {results['summary']['silhouette_score']:.4f}")
    print(f"Normalized Mutual Information: {results['summary']['normalized_mutual_info']:.4f}")
    print(f"Mean Cluster Purity: {results['summary']['mean_purity']:.4f}")
