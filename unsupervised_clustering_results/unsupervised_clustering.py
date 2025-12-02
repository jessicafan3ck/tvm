import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")


class AestheticFeatureExtractor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.raw_features = []
        self.metadata = []
        self.labels = []
        self.feature_df = None
        
    def load_all_files(self):
        json_files = sorted(list(self.data_dir.glob('*.json')))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                self.raw_features.append(data['features'])
                self.metadata.append(data['meta'])
                
                filename = json_file.stem
                parts = filename.split('_', 1)
                label = parts[1] if len(parts) == 2 else filename
                self.labels.append(label)
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return self._flatten_to_dataframe()
    
    def _flatten_to_dataframe(self):
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
        
    def extract_objective_features(self):
        candidate_cols = self._get_all_feature_names()
        available_cols = [col for col in candidate_cols if col in self.feature_df.columns]

        if not available_cols:
            raise ValueError("No objective features found.")
        
        self.objective_features = self.feature_df[available_cols].fillna(0)
        self.scaled_features = self.scaler.fit_transform(self.objective_features)
        
        print(f"Extracted {self.objective_features.shape[1]} objective features")
        return self.scaled_features
    
    def _get_all_feature_names(self):
        all_features = []
        for category_name, category_features in self.FEATURE_CATEGORIES.items():
            for f in category_features:
                all_features.append(f"{category_name}_{f}")
        return all_features


class UnsupervisedClusterer:
    def __init__(self, scaled_features: np.ndarray):
        self.scaled_features = scaled_features
        self.results = {}
        
    def find_optimal_k(self, k_range: range = range(2, 16)):
        print(f"\nTesting k from {min(k_range)} to {max(k_range)}...")
        
        metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in k_range:
            print(f"  k={k}...", end=" ")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_features)
            
            sil = silhouette_score(self.scaled_features, labels)
            db = davies_bouldin_score(self.scaled_features, labels)
            ch = calinski_harabasz_score(self.scaled_features, labels)
            
            metrics['silhouette'].append(sil)
            metrics['davies_bouldin'].append(db)
            metrics['calinski_harabasz'].append(ch)
            
            print(f"Silhouette={sil:.4f}")
        
        # Find optimal k by each metric
        optimal_k_sil = k_range[np.argmax(metrics['silhouette'])]
        optimal_k_db = k_range[np.argmin(metrics['davies_bouldin'])]
        optimal_k_ch = k_range[np.argmax(metrics['calinski_harabasz'])]
        
        self.results = {
            'k_range': list(k_range),
            'silhouette': metrics['silhouette'],
            'davies_bouldin': metrics['davies_bouldin'],
            'calinski_harabasz': metrics['calinski_harabasz'],
            'optimal_k_silhouette': optimal_k_sil,
            'optimal_k_davies_bouldin': optimal_k_db,
            'optimal_k_calinski_harabasz': optimal_k_ch
        }
        
        print(f"\nOptimal k by Silhouette: {optimal_k_sil} (score: {metrics['silhouette'][optimal_k_sil-min(k_range)]:.4f})")
        print(f"Optimal k by Davies-Bouldin: {optimal_k_db} (score: {metrics['davies_bouldin'][optimal_k_db-min(k_range)]:.4f})")
        print(f"Optimal k by Calinski-Harabasz: {optimal_k_ch} (score: {metrics['calinski_harabasz'][optimal_k_ch-min(k_range)]:.4f})")
        
        return self.results
    
    def cluster_with_k(self, k: int):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.scaled_features)
        return labels


class Visualizer:
    @staticmethod
    def plot_clustering_metrics(results: Dict, figsize: Tuple = (15, 5)):
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        k_range = results['k_range']
        
        axes[0].plot(k_range, results['silhouette'], 'o-', linewidth=2.5, markersize=8, color='#FF6B6B')
        axes[0].axvline(results['optimal_k_silhouette'], color='green', linestyle='--', linewidth=2.5, 
                       label=f"Optimal k={results['optimal_k_silhouette']}")
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Silhouette Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0].set_title('Silhouette Analysis', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        axes[0].axhline(0, color='red', linestyle=':', alpha=0.5)
        
        axes[1].plot(k_range, results['davies_bouldin'], 'o-', linewidth=2.5, markersize=8, color='#4ECDC4')
        axes[1].axvline(results['optimal_k_davies_bouldin'], color='green', linestyle='--', linewidth=2.5,
                       label=f"Optimal k={results['optimal_k_davies_bouldin']}")
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
        axes[1].set_title('Davies-Bouldin Analysis', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        
        axes[2].plot(k_range, results['calinski_harabasz'], 'o-', linewidth=2.5, markersize=8, color='#95E1D3')
        axes[2].axvline(results['optimal_k_calinski_harabasz'], color='green', linestyle='--', linewidth=2.5,
                       label=f"Optimal k={results['optimal_k_calinski_harabasz']}")
        axes[2].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Calinski-Harabasz Index (Higher is Better)', fontsize=12, fontweight='bold')
        axes[2].set_title('Calinski-Harabasz Analysis', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=10)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_umap_clustering(umap_features: np.ndarray, labels: np.ndarray, 
                            title: str = "Unsupervised Clusters in UMAP Space",
                            figsize: Tuple = (14, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            umap_features[:, 0], umap_features[:, 1],
            c=labels, cmap='tab10', s=20, alpha=0.6, edgecolors='none'
        )
        
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel('UMAP Component 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP Component 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cluster_composition(feature_df: pd.DataFrame, labels: np.ndarray,
                                k: int, figsize: Tuple = (16, 10)):
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        cluster_df = feature_df.copy()
        cluster_df['cluster'] = labels
        
        modifier_counts = pd.crosstab(cluster_df['cluster'], cluster_df['base_modifier'])
        modifier_counts.plot(kind='bar', stacked=True, ax=axes[0], 
                            colormap='tab20', width=0.7)
        axes[0].set_title(f'Cluster Composition by Base Modifier (k={k})', 
                         fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Cluster', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        axes[0].legend(title='Modifier', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        cluster_sizes = cluster_df['cluster'].value_counts().sort_index()
        axes[1].bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1].set_title(f'Cluster Sizes (k={k})', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Cluster', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(cluster_sizes.values):
            axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cluster_modifier_heatmap(feature_df: pd.DataFrame, labels: np.ndarray,
                                     k: int, figsize: Tuple = (12, 8)):
        cluster_df = feature_df.copy()
        cluster_df['cluster'] = labels
        
        modifier_counts = pd.crosstab(cluster_df['cluster'], cluster_df['base_modifier'])
        modifier_pct = modifier_counts.div(modifier_counts.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(modifier_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Percentage (%)'}, ax=ax, linewidths=0.5)
        
        ax.set_title(f'Modifier Distribution Across Clusters (k={k})\nPercentage within each cluster', 
                    fontsize=13, fontweight='bold', pad=20)
        ax.set_xlabel('Base Modifier', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cluster', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        return fig


def main():
    data_dir = (
        '/Users/jessicafan/Downloads/'
        'tvm board - Nov 22nd 2025 (5060 images)/'
        'TVM JSON'
    )
    output_dir = Path('./unsupervised_clustering_results')
    output_dir.mkdir(exist_ok=True)
    
    print("\n[1/6] Loading all samples...")
    extractor = AestheticFeatureExtractor(data_dir)
    feature_df = extractor.load_all_files()
    print(f"Loaded {len(feature_df)} samples")
    print(f"Unique modifiers: {feature_df['base_modifier'].nunique()}")
    
    print("\n[2/6] Extracting objective features (F_obj)...")
    obj_analyzer = ObjectiveFeatureAnalyzer(feature_df)
    scaled_features = obj_analyzer.extract_objective_features()
    
    print("\n[3/6] Reducing to UMAP for visualization...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
    umap_features = reducer.fit_transform(scaled_features)
    print(f"UMAP complete")
    
    print("\n[4/6] Finding optimal number of clusters...")
    clusterer = UnsupervisedClusterer(scaled_features)
    results = clusterer.find_optimal_k(k_range=range(2, 16))

    print("\n[5/6] Creating visualizations...")
    fig1 = Visualizer.plot_clustering_metrics(results)
    fig1.savefig(output_dir / 'clustering_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("  ✓ Clustering metrics saved")
    
    print("\n  Testing k=5 (the hypothesized structure)...")
    labels_k5 = clusterer.cluster_with_k(5)
    
    fig2 = Visualizer.plot_umap_clustering(umap_features, labels_k5, 
                                          title="5-Group Unsupervised Clustering in UMAP Space")
    fig2.savefig(output_dir / 'umap_k5.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("  ✓ UMAP k=5 visualization saved")
    
    fig3 = Visualizer.plot_cluster_composition(feature_df, labels_k5, k=5)
    fig3.savefig(output_dir / 'cluster_composition_k5.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("  ✓ Cluster composition (k=5) saved")
    
    fig4 = Visualizer.plot_cluster_modifier_heatmap(feature_df, labels_k5, k=5)
    fig4.savefig(output_dir / 'modifier_heatmap_k5.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("  ✓ Modifier heatmap (k=5) saved")
    
    optimal_k = results['optimal_k_silhouette']
    if optimal_k != 5:
        print(f"\n  Testing optimal k={optimal_k}...")
        labels_optimal = clusterer.cluster_with_k(optimal_k)
        
        fig5 = Visualizer.plot_umap_clustering(umap_features, labels_optimal,
                                              title=f"Optimal Clustering (k={optimal_k}) in UMAP Space")
        fig5.savefig(output_dir / f'umap_k{optimal_k}.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print(f"  ✓ UMAP k={optimal_k} visualization saved")
        
        fig6 = Visualizer.plot_cluster_composition(feature_df, labels_optimal, k=optimal_k)
        fig6.savefig(output_dir / f'cluster_composition_k{optimal_k}.png', dpi=300, bbox_inches='tight')
        plt.close(fig6)
        print(f"  ✓ Cluster composition (k={optimal_k}) saved")
        
        fig7 = Visualizer.plot_cluster_modifier_heatmap(feature_df, labels_optimal, k=optimal_k)
        fig7.savefig(output_dir / f'modifier_heatmap_k{optimal_k}.png', dpi=300, bbox_inches='tight')
        plt.close(fig7)
        print(f"  ✓ Modifier heatmap (k={optimal_k}) saved")
    
    print("\n[6/6] Generating summary...")
    
    summary = {
        'total_samples': len(feature_df),
        'total_modifiers': int(feature_df['base_modifier'].nunique()),
        'clustering_metrics': {
            'k_range': results['k_range'],
            'silhouette_scores': [float(x) for x in results['silhouette']],
            'davies_bouldin_scores': [float(x) for x in results['davies_bouldin']],
            'calinski_harabasz_scores': [float(x) for x in results['calinski_harabasz']],
            'optimal_k_silhouette': int(results['optimal_k_silhouette']),
            'optimal_k_davies_bouldin': int(results['optimal_k_davies_bouldin']),
            'optimal_k_calinski_harabasz': int(results['optimal_k_calinski_harabasz'])
        }
    }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nTotal Samples: {len(feature_df)}")
    print(f"Base Modifiers: {feature_df['base_modifier'].unique().tolist()}")
    print(f"\nOptimal Clustering:")
    print(f"  By Silhouette:        k = {results['optimal_k_silhouette']} (score: {results['silhouette'][results['optimal_k_silhouette']-2]:.4f})")
    print(f"  By Davies-Bouldin:    k = {results['optimal_k_davies_bouldin']} (score: {results['davies_bouldin'][results['optimal_k_davies_bouldin']-2]:.4f})")
    print(f"  By Calinski-Harabasz: k = {results['optimal_k_calinski_harabasz']} (score: {results['calinski_harabasz'][results['optimal_k_calinski_harabasz']-2]:.4f})")
    
    print(f"\nHypothesized k=5:")
    labels_k5 = clusterer.cluster_with_k(5)
    sil_k5 = silhouette_score(scaled_features, labels_k5)
    print(f"  Silhouette score: {sil_k5:.4f}")
    
    print(f"\nResults saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
