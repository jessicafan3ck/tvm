import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
)

try:
    import clip
    import torch
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("WARNING: CLIP not available. Install with: pip install openai-clip torch")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib import cm

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

AESTHETIC_SENTIMENT_GROUPS = {
    'Whimsical/Light': ['whimsical'],
    'Nostalgic/Temporal': ['nostalgic'],
    'Candid/Authentic': ['candid'],
    'Ethereal/Transcendent': ['ethereal'],
    'Haunted/Ominous': ['haunted'],
    'Introspective/Thoughtful': ['introspective'],
    'Melancholic/Somber': ['melancholic'],
    'Radiant/Joyful': ['radiant'],
    'Romantic/Intimate': ['romantic']
}


class CLIPSemanticExtractor:
    
    def __init__(self, model_name: str = "ViT-B/32"):
        if not HAS_CLIP:
            raise ImportError("CLIP not available. Install: pip install openai-clip torch")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.embedding_cache = {}
        print(f"CLIP loaded on {self.device}")
    
    def embed_text(self, text: str) -> np.ndarray:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        with torch.no_grad():
            text_tokens = clip.tokenize(text).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)
            embedding = text_embedding.cpu().numpy()[0]
            self.embedding_cache[text] = embedding
            return embedding
    
    def embed_modifiers(self, modifiers: List[str]) -> np.ndarray:
        embeddings = []
        for modifier in modifiers:
            embedding = self.embed_text(modifier)
            embeddings.append(embedding)
        return np.array(embeddings)


class AestheticFeatureExtractor:

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.raw_features = []
        self.metadata = []
        self.labels = []
        self.feature_df = None
        
    def load_all_files(self) -> pd.DataFrame:
        json_files = sorted(list(self.data_dir.glob('*.json')))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                self.raw_features.append(data['features'])
                self.metadata.append(data['meta'])
                
                # Extract label from filename (format: number_[label].json)
                filename = json_file.stem
                parts = filename.split('_', 1)
                if len(parts) == 2:
                    label = parts[1]
                else:
                    label = filename
                self.labels.append(label)
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return self._flatten_to_dataframe()
    
    def _flatten_to_dataframe(self) -> pd.DataFrame:
        flattened = []
        
        for meta, features, label in zip(self.metadata, self.raw_features, self.labels):
            row = {
                '_filename': meta['_filename'],
                'label': label,
                'base_modifier': label.split()[0] 
            }
            
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
        print(f"Unique base modifiers: {len(self.feature_df['base_modifier'].unique())}")
        print(f"Base modifiers: {sorted(self.feature_df['base_modifier'].unique())}")
        return self.feature_df


class ObjectiveFeatureAnalyzer:
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
        self.scaler = RobustScaler()
        self.scaled_features = None
        
    def extract_objective_features(self) -> np.ndarray:
        candidate_cols = self._get_all_feature_names()
        available_cols = [col for col in candidate_cols if col in self.feature_df.columns]

        if not available_cols:
            print("\n[ERROR] No objective feature columns found.")
            raise ValueError("No objective features found.")
        
        self.objective_features = self.feature_df[available_cols].fillna(0)
        self.scaled_features = self.scaler.fit_transform(self.objective_features)
        
        print(f"Extracted {self.objective_features.shape[1]} objective features (F_obj)")
        return self.scaled_features
    
    def _get_all_feature_names(self) -> List[str]:
        all_features = []
        for category_name, category_features in self.FEATURE_CATEGORIES.items():
            for f in category_features:
                all_features.append(f"{category_name}_{f}")
        return all_features
    
    def feature_importance_by_category(self) -> Dict[str, float]:
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
                    list(self.objective_features.columns).index(f) for f in available
                ]
                variance_by_category[category] = np.mean(
                    pca.explained_variance_ratio_[feature_indices]
                )
        
        return variance_by_category


class DimensionalityReducer:
    
    def __init__(self, scaled_features: np.ndarray, n_neighbors: int = 15):
        self.scaled_features = scaled_features
        self.n_neighbors = n_neighbors
        self.reductions = {}
        
    def apply_pca(self, n_components: int = 50) -> Tuple[np.ndarray, float]:
        max_components = min(self.scaled_features.shape[0], self.scaled_features.shape[1])
        if n_components > max_components:
            n_components = max_components

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(self.scaled_features)
        explained_variance = pca.explained_variance_ratio_.sum()
        self.reductions['pca'] = {'model': pca, 'data': reduced, 'variance': explained_variance}
        print(f"PCA: {n_components} components explain {explained_variance:.2%} variance")
        return reduced, explained_variance
    
    def apply_umap(self, n_components: int = 20, metric: str = 'euclidean') -> np.ndarray:
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


class ModifierDisentanglement:

    def __init__(self, feature_df: pd.DataFrame, scaled_objective_features: np.ndarray):
        self.feature_df = feature_df
        self.scaled_objective_features = scaled_objective_features
        self.modifier_aggregates = {}
        self.modifier_features = None
        
    def aggregate_by_modifier(self) -> Tuple[np.ndarray, List[str]]:

        modifiers = sorted(self.feature_df['base_modifier'].unique())
        aggregated = []
        
        for modifier in modifiers:
            mask = self.feature_df['base_modifier'] == modifier
            modifier_features = self.scaled_objective_features[mask]
            # Average across all subject variations
            mean_features = modifier_features.mean(axis=0)
            aggregated.append(mean_features)
            self.modifier_aggregates[modifier] = {
                'mean': mean_features,
                'std': modifier_features.std(axis=0),
                'n_samples': mask.sum(),
                'std_across_samples': modifier_features.std()
            }
        
        self.modifier_features = np.array(aggregated)
        print(f"\nAggregated {len(modifiers)} base modifiers (F_obj averaged across subjects)")
        return self.modifier_features, modifiers
    
    def compute_modifier_dispersion(self) -> Dict:

        dispersion_stats = {}
        for modifier, stats in self.modifier_aggregates.items():
            dispersion_stats[modifier] = {
                'std_across_subjects': float(stats['std_across_samples']),
                'n_subjects': stats['n_samples'],
                'feature_variance': float(stats['std'].mean())
            }
        return dispersion_stats
    
    def analyze_subject_modifier_interactions(self) -> Dict:

        interactions = {}
        total_variance = np.var(self.scaled_objective_features)
        
        # Variance within modifiers (subject effects)
        within_modifier_variance = 0
        for modifier in self.feature_df['base_modifier'].unique():
            mask = self.feature_df['base_modifier'] == modifier
            within_modifier_variance += np.var(self.scaled_objective_features[mask])
        within_modifier_variance /= len(self.feature_df['base_modifier'].unique())
        
        # Variance between modifiers
        between_modifier_variance = total_variance - within_modifier_variance
        
        return {
            'total_variance': float(total_variance),
            'within_modifier_variance': float(within_modifier_variance),
            'between_modifier_variance': float(between_modifier_variance),
            'modifier_variance_ratio': float(between_modifier_variance / total_variance),
            'subject_variance_ratio': float(within_modifier_variance / total_variance)
        }


class SentimentModifierClustering:
    def __init__(self, modifier_features: np.ndarray, modifiers: List[str],
                 semantic_embeddings: np.ndarray = None):
        self.modifier_features = modifier_features
        self.modifiers = modifiers
        self.semantic_embeddings = semantic_embeddings
        self.scaler = StandardScaler()
        self.clusters = {}
        
    def cluster_by_sentiment(self, n_clusters_range: range = range(2, 10)) -> Dict:
        features_to_cluster = self.semantic_embeddings if self.semantic_embeddings is not None else self.modifier_features
        scaled_features = self.scaler.fit_transform(features_to_cluster)
        
        n_modifiers = scaled_features.shape[0]
        n_clusters_range = range(2, min(n_modifiers, max(n_clusters_range)+1))
        
        results = {}
        
        kmeans_metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in n_clusters_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_features)
            
            kmeans_metrics['silhouette'].append(silhouette_score(scaled_features, labels))
            kmeans_metrics['davies_bouldin'].append(davies_bouldin_score(scaled_features, labels))
            kmeans_metrics['calinski_harabasz'].append(calinski_harabasz_score(scaled_features, labels))
        
        optimal_k = n_clusters_range[np.argmax(kmeans_metrics['silhouette'])]
        
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans_final.fit_predict(scaled_features)
        
        results['kmeans'] = {
            'labels': kmeans_labels,
            'optimal_k': optimal_k,
            'silhouette_scores': kmeans_metrics['silhouette'],
            'davies_bouldin_scores': kmeans_metrics['davies_bouldin'],
            'calinski_harabasz_scores': kmeans_metrics['calinski_harabasz'],
            'k_range': list(n_clusters_range)
        }
        
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        hierarchical_labels = hierarchical.fit_predict(scaled_features)
        results['hierarchical'] = {
            'labels': hierarchical_labels,
            'optimal_k': optimal_k
        }
        
        if n_modifiers > 3:
            try:
                n_neighbors = min(4, n_modifiers - 1)
                spectral = SpectralClustering(
                    n_clusters=optimal_k, 
                    random_state=42, 
                    affinity='nearest_neighbors',
                    n_neighbors=n_neighbors
                )
                spectral_labels = spectral.fit_predict(scaled_features)
                results['spectral'] = {
                    'labels': spectral_labels,
                    'optimal_k': optimal_k
                }
            except Exception as e:
                print(f"  Spectral clustering skipped: {e}")
        
        self.clusters = results
        
        print(f"\nSentiment Clustering Results:")
        print(f"  Number of modifiers: {n_modifiers}")
        print(f"  Optimal k: {optimal_k}")
        print(f"  Silhouette score: {kmeans_metrics['silhouette'][optimal_k-min(n_clusters_range)]:.4f}")
        
        return results
    
    def get_sentiment_groups(self) -> Dict[int, List[str]]:
        if 'kmeans' not in self.clusters:
            return {}
        
        labels = self.clusters['kmeans']['labels']
        groups = {}
        for modifier, label in zip(self.modifiers, labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(modifier)
        
        return groups
    
    def analyze_sentiment_structure(self) -> Dict:
        if 'kmeans' not in self.clusters:
            return {}
        
        labels = self.clusters['kmeans']['labels']
        groups = self.get_sentiment_groups()
        
        analysis = {
            'n_sentiment_groups': len(groups),
            'groups': groups,
            'group_sizes': {str(k): len(v) for k, v in groups.items()}
        }
        
        return analysis


class AestheticVisualization:
    @staticmethod
    def plot_dimensionality_reduction(reduced_features: np.ndarray, labels: np.ndarray = None,
                                      label_names: List[str] = None,
                                      title: str = "Dimensionality Reduction", 
                                      figsize: Tuple = (14, 10)):
        """Plot 2D or 3D dimensionality reduction with optional label coloring."""
        fig = plt.figure(figsize=figsize)
        
        numeric_labels = None
        if labels is not None:
            if isinstance(labels[0], str):
                unique_labels = np.unique(labels)
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_to_idx[label] for label in labels])
            else:
                numeric_labels = labels
        
        if reduced_features.shape[1] >= 3:
            ax = fig.add_subplot(111, projection='3d')
            if numeric_labels is not None:
                scatter = ax.scatter(
                    reduced_features[:, 0], reduced_features[:, 1],
                    reduced_features[:, 2], c=numeric_labels, cmap='tab20', s=30, alpha=0.6
                )
                plt.colorbar(scatter, ax=ax, label='Modifier')
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
            if numeric_labels is not None:
                scatter = ax.scatter(
                    reduced_features[:, 0], reduced_features[:, 1],
                    c=numeric_labels, cmap='tab20', s=30, alpha=0.7
                )
                plt.colorbar(scatter, ax=ax, label='Modifier')
            else:
                ax.scatter(reduced_features[:, 0], reduced_features[:, 1], s=30, alpha=0.6)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_modifier_latent_space(reduced_features: np.ndarray, modifiers: List[str],
                                   sentiment_labels: np.ndarray = None,
                                   title: str = "Modifier Latent Space (F_obj)",
                                   figsize: Tuple = (14, 10)):
        fig = plt.figure(figsize=figsize)
        
        if reduced_features.shape[1] >= 3:
            ax = fig.add_subplot(111, projection='3d')
            if sentiment_labels is not None:
                scatter = ax.scatter(
                    reduced_features[:, 0], reduced_features[:, 1],
                    reduced_features[:, 2], c=sentiment_labels, cmap='tab10', s=200, alpha=0.7
                )
                plt.colorbar(scatter, ax=ax, label='Sentiment Group')
            else:
                ax.scatter(reduced_features[:, 0], reduced_features[:, 1],
                          reduced_features[:, 2], s=200, alpha=0.7)
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        else:
            ax = fig.add_subplot(111)
            if sentiment_labels is not None:
                scatter = ax.scatter(
                    reduced_features[:, 0], reduced_features[:, 1],
                    c=sentiment_labels, cmap='tab10', s=300, alpha=0.7, edgecolors='black', linewidth=1.5
                )
                plt.colorbar(scatter, ax=ax, label='Sentiment Group')
            else:
                ax.scatter(reduced_features[:, 0], reduced_features[:, 1],
                          s=300, alpha=0.7, edgecolors='black', linewidth=1.5)
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        # Annotate modifiers
        for i, modifier in enumerate(modifiers):
            ax.text(reduced_features[i, 0], reduced_features[i, 1],
                   f'  {modifier}', fontsize=9, ha='left')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sentiment_clustering_metrics(clustering_results: Dict, figsize: Tuple = (15, 5)):
        if 'kmeans' not in clustering_results:
            return None
        
        kmeans = clustering_results['kmeans']
        k_range = kmeans['k_range']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].plot(k_range, kmeans['silhouette_scores'], 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Sentiment Groups (k)')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Sentiment Group Silhouette Analysis')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(kmeans['optimal_k'], color='r', linestyle='--', alpha=0.7, linewidth=2)
        
        axes[1].plot(k_range, kmeans['davies_bouldin_scores'], 'o-', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Number of Sentiment Groups (k)')
        axes[1].set_ylabel('Davies-Bouldin Index')
        axes[1].set_title('Davies-Bouldin Analysis (Lower is Better)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(kmeans['optimal_k'], color='r', linestyle='--', alpha=0.7, linewidth=2)
        
        axes[2].plot(k_range, kmeans['calinski_harabasz_scores'], 'o-', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Number of Sentiment Groups (k)')
        axes[2].set_ylabel('Calinski-Harabasz Index')
        axes[2].set_title('Calinski-Harabasz Analysis')
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(kmeans['optimal_k'], color='r', linestyle='--', alpha=0.7, linewidth=2)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sentiment_groups(sentiment_groups: Dict[int, List[str]], figsize: Tuple = (12, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        
        groups_sorted = sorted(sentiment_groups.items())
        y_pos = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups_sorted)))
        
        for (group_id, modifiers), color in zip(groups_sorted, colors):
            group_text = f"Group {group_id}: {', '.join(modifiers)}"
            ax.text(0.05, y_pos, group_text, fontsize=11, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            y_pos -= 0.08
        
        ax.set_xlim(0, 1)
        ax.set_ylim(y_pos - 0.05, 0.1)
        ax.axis('off')
        ax.set_title('Semantic Sentiment Groups', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_subject_modifier_interaction(interaction_stats: Dict, figsize: Tuple = (10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        
        modifier_var = max(0, interaction_stats['between_modifier_variance'])
        subject_var = max(0, interaction_stats['within_modifier_variance'])
        
        if modifier_var + subject_var == 0:
            modifier_var = 1
            subject_var = 1
        
        variances = [modifier_var, subject_var]
        labels = ['Modifier Effect\n(Aesthetic Sentiment)', 'Subject Effect\n(Matter Variance)']
        colors = ['#FF6B6B', '#4ECDC4']
        
        wedges, texts, autotexts = ax.pie(
            variances, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        ax.set_title('Variance Decomposition: Modifier vs Subject', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_modifier_dispersion(dispersion_stats: Dict, figsize: Tuple = (14, 8)):
        modifiers = sorted(dispersion_stats.keys())
        dispersions = [dispersion_stats[m]['std_across_subjects'] for m in modifiers]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(modifiers)))
        bars = ax.barh(modifiers, dispersions, color=colors)
        
        ax.set_xlabel('Within-Modifier Dispersion (Higher = More Subject Variation)', fontsize=11)
        ax.set_title('Subject Variation Within Each Aesthetic Modifier', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig


class EnhancedAestheticPipeline:
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.results = {}
        
    def run_full_analysis(self, use_clip: bool = True, n_pca_components: int = 50,
                         n_umap_components: int = 20):
        
        print("=" * 80)
        print("ENHANCED AESTHETIC LATENT SPACE ANALYSIS")
        print("=" * 80)
        
        print("\n[1/9] Extracting Features from Labeled JSONs...")
        extractor = AestheticFeatureExtractor(self.data_dir)
        feature_df = extractor.load_all_files()
        
        print("\n[2/9] Analyzing Objective Features (F_obj)...")
        obj_analyzer = ObjectiveFeatureAnalyzer(feature_df)
        scaled_obj_features = obj_analyzer.extract_objective_features()
        
        feature_importance = obj_analyzer.feature_importance_by_category()
        self.results['feature_importance'] = feature_importance
        print("Feature Importance by Category:")
        for cat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {imp:.4f}")
        
        print("\n[3/9] Extracting Semantic Embeddings (F_sem via CLIP)...")
        semantic_embeddings = None
        if use_clip and HAS_CLIP:
            try:
                clip_extractor = CLIPSemanticExtractor()
                unique_modifiers = sorted(feature_df['base_modifier'].unique())
                semantic_embeddings = clip_extractor.embed_modifiers(unique_modifiers)
                print(f"Generated {len(unique_modifiers)} CLIP embeddings (512-dim)")
                self.results['semantic_embeddings_available'] = True
            except Exception as e:
                print(f"CLIP extraction failed: {e}")
                self.results['semantic_embeddings_available'] = False
        else:
            print("CLIP not available or disabled")
            self.results['semantic_embeddings_available'] = False

        print("\n[4/9] Applying Dimensionality Reduction (F_obj)...")
        reducer = DimensionalityReducer(scaled_obj_features, n_neighbors=15)
        
        pca_reduced, pca_variance = reducer.apply_pca(n_components=n_pca_components)
        umap_reduced = reducer.apply_umap(n_components=n_umap_components)
        tsne_reduced = reducer.apply_tsne(n_components=2)
        
        self.results['pca_variance_explained'] = float(pca_variance)

        print("\n[5/9] Analyzing Modifier-Subject Disentanglement...")
        disentanglement = ModifierDisentanglement(feature_df, scaled_obj_features)
        modifier_features, modifiers = disentanglement.aggregate_by_modifier()
        
        dispersion_stats = disentanglement.compute_modifier_dispersion()
        self.results['modifier_dispersion'] = dispersion_stats
        
        interaction_stats = disentanglement.analyze_subject_modifier_interactions()
        self.results['subject_modifier_interaction'] = interaction_stats
        
        print(f"Modifier Variance Ratio: {interaction_stats['modifier_variance_ratio']:.4f}")
        print(f"Subject Variance Ratio: {interaction_stats['subject_variance_ratio']:.4f}")
        print(f"  → {interaction_stats['modifier_variance_ratio']*100:.1f}% of variance from modifiers")
        print(f"  → {interaction_stats['subject_variance_ratio']*100:.1f}% of variance from subject/environment")

        print("\n[6/9] Reducing Modifier-Only Latent Space...")
        modifier_reducer = DimensionalityReducer(modifier_features, n_neighbors=min(5, len(modifiers)-1))
        modifier_umap = modifier_reducer.apply_umap(n_components=2)
        modifier_pca = modifier_reducer.apply_pca(n_components=min(10, len(modifiers)-1))

        print("\n[7/9] Clustering Modifiers by Semantic Sentiment...")
        sentiment_clusterer = SentimentModifierClustering(
            modifier_features, modifiers,
            semantic_embeddings=semantic_embeddings
        )
        clustering_results = sentiment_clusterer.cluster_by_sentiment(n_clusters_range=range(2, min(10, len(modifiers))))
        self.results['sentiment_clustering'] = clustering_results
        
        sentiment_structure = sentiment_clusterer.analyze_sentiment_structure()
        self.results['sentiment_groups'] = sentiment_structure
        
        print("\nSentiment Group Composition:")
        for group_id, group_modifiers in sorted(sentiment_structure['groups'].items()):
            print(f"  Group {group_id}: {group_modifiers}")
        
        print("\n[8/9] Generating Visualizations...")
        viz_results = self._create_visualizations(
            reduced_features=umap_reduced,
            feature_df=feature_df,
            modifier_features=modifier_features,
            modifier_umap=modifier_umap,
            modifiers=modifiers,
            sentiment_labels=clustering_results['kmeans']['labels'],
            clustering_results=clustering_results,
            sentiment_structure=sentiment_structure,
            interaction_stats=interaction_stats,
            dispersion_stats=dispersion_stats
        )
        self.results['visualizations'] = viz_results
        
        print("\n[9/9] Generating Summary Report...")
        summary = self._generate_summary(
            feature_df, modifier_features, clustering_results,
            sentiment_structure, interaction_stats
        )
        self.results['summary'] = summary
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE - KEY FINDINGS:")
        print("=" * 80)
        print(f"Aesthetic Modifiers: {len(modifiers)}")
        print(f"Sentiment Groupings: {sentiment_structure['n_sentiment_groups']}")
        print(f"Modifier Variance: {interaction_stats['modifier_variance_ratio']*100:.1f}%")
        print(f"Best Separation: k={clustering_results['kmeans']['optimal_k']} "
              f"(silhouette: {clustering_results['kmeans']['silhouette_scores'][clustering_results['kmeans']['optimal_k']-2]:.4f})")
        
        return self.results
    
    def _create_visualizations(self, **kwargs) -> Dict:
        viz_dict = {}
        
        print("  - Creating full latent space visualization...")
        fig1 = AestheticVisualization.plot_dimensionality_reduction(
            kwargs['reduced_features'],
            labels=kwargs['feature_df']['base_modifier'].values,
            title="Full Dataset Latent Space (F_obj + subjects)",
            figsize=(14, 10)
        )
        viz_dict['full_latent_space'] = fig1
        
        print("  - Creating modifier-only latent space visualization...")
        fig2 = AestheticVisualization.plot_modifier_latent_space(
            kwargs['modifier_umap'],
            kwargs['modifiers'],
            sentiment_labels=kwargs['sentiment_labels'],
            title="Modifier Latent Space (F_obj aggregated by aesthetic modifier)",
            figsize=(14, 10)
        )
        viz_dict['modifier_latent_space'] = fig2
        
        print("  - Creating sentiment clustering metrics...")
        fig3 = AestheticVisualization.plot_sentiment_clustering_metrics(
            kwargs['clustering_results']
        )
        if fig3:
            viz_dict['sentiment_clustering_metrics'] = fig3
        
        print("  - Creating sentiment groups visualization...")
        fig4 = AestheticVisualization.plot_sentiment_groups(
            kwargs['sentiment_structure']['groups']
        )
        viz_dict['sentiment_groups'] = fig4
        
        print("  - Creating variance decomposition visualization...")
        fig5 = AestheticVisualization.plot_subject_modifier_interaction(
            kwargs['interaction_stats']
        )
        viz_dict['variance_decomposition'] = fig5

        print("  - Creating modifier dispersion visualization...")
        fig6 = AestheticVisualization.plot_modifier_dispersion(
            kwargs['dispersion_stats']
        )
        viz_dict['modifier_dispersion'] = fig6
        
        return viz_dict
    
    def _generate_summary(self, feature_df, modifier_features, clustering_results,
                         sentiment_structure, interaction_stats) -> Dict:
        return {
            'total_samples': len(feature_df),
            'n_modifiers': len(modifier_features),
            'n_sentiment_groups': sentiment_structure['n_sentiment_groups'],
            'modifier_variance_ratio': float(interaction_stats['modifier_variance_ratio']),
            'subject_variance_ratio': float(interaction_stats['subject_variance_ratio']),
            'optimal_sentiment_k': clustering_results['kmeans']['optimal_k'],
            'sentiment_silhouette': float(
                clustering_results['kmeans']['silhouette_scores'][
                    clustering_results['kmeans']['optimal_k'] - min(clustering_results['kmeans']['k_range'])
                ]
            )
        }
    
    def save_results(self, output_dir: str = './aesthetic_analysis_enhanced_results'):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            k: v for k, v in self.results.items()
            if k not in ['visualizations', 'sentiment_clustering']
        }
        
        def convert_to_serializable(obj):
            """Convert numpy/non-serializable types to JSON-serializable types."""
            if isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        summary_data = convert_to_serializable(summary_data)
        
        with open(output_path / 'analysis_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        for name, fig in self.results.get('visualizations', {}).items():
            if fig:
                fig.savefig(output_path / f'{name}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    data_directory = (
        '/Users/jessicafan/Downloads/'
        'tvm board - Nov 22nd 2025 (5060 images)/'
        'TVM JSON'
    )
    
    pipeline = EnhancedAestheticPipeline(data_directory)
    results = pipeline.run_full_analysis(
        use_clip=True,  
        n_pca_components=50,
        n_umap_components=20
    )
    
    pipeline.save_results('./aesthetic_analysis_enhanced_results')
    
    print("\n\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print(f"Modifiers: {results['summary']['n_modifiers']}")
    print(f"Sentiment Groups Found: {results['summary']['n_sentiment_groups']}")
    print(f"Modifier Variance: {results['summary']['modifier_variance_ratio']*100:.1f}%")
    print(f"Subject Variance: {results['summary']['subject_variance_ratio']*100:.1f}%")
