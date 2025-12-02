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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        return self.scaled_features
    
    def _get_all_feature_names(self):
        all_features = []
        for category_name, category_features in self.FEATURE_CATEGORIES.items():
            for f in category_features:
                all_features.append(f"{category_name}_{f}")
        return all_features
    
    def get_feature_importance(self, labels: np.ndarray) -> Dict[str, float]:
        importance = {}
        
        for col in self.objective_features.columns:
            values = self.objective_features[col].values
            cluster_0_mean = values[labels == 0].mean()
            cluster_1_mean = values[labels == 1].mean()
            
            std = values.std()
            if std > 0:
                diff = abs(cluster_0_mean - cluster_1_mean) / std
            else:
                diff = 0
            
            importance[col] = {
                'difference': diff,
                'cluster_0_mean': cluster_0_mean,
                'cluster_1_mean': cluster_1_mean,
                'ratio': cluster_1_mean / cluster_0_mean if cluster_0_mean > 0 else 0
            }
        
        return importance


class ClusterAnalyzer:
    
    @staticmethod
    def cluster_data(scaled_features: np.ndarray, k: int = 2):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        
        sil = silhouette_score(scaled_features, labels)
        db = davies_bouldin_score(scaled_features, labels)
        ch = calinski_harabasz_score(scaled_features, labels)
        
        return labels, {'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch}
    
    @staticmethod
    def analyze_cluster_composition(feature_df: pd.DataFrame, labels: np.ndarray) -> Dict:
        cluster_df = feature_df.copy()
        cluster_df['cluster'] = labels
        
        composition = {}
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            composition[cluster_id] = {
                'size': mask.sum(),
                'percentage': 100 * mask.sum() / len(labels),
                'modifiers': cluster_df[mask]['base_modifier'].value_counts().to_dict(),
                'labels': cluster_df[mask]['label'].value_counts().head(10).to_dict()
            }
        
        return composition


class Visualizer:
    
    @staticmethod
    def plot_feature_importance(importance: Dict[str, float], top_n: int = 20, 
                                figsize: Tuple = (12, 8)):
        sorted_features = sorted(importance.items(), 
                                key=lambda x: x[1]['difference'], 
                                reverse=True)
        
        top_features = sorted_features[:top_n]
        feature_names = [f[0].replace('_', ' ').title()[:30] for f in top_features]
        differences = [f[1]['difference'] for f in top_features]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#FF6B6B' if d > 0 else '#4ECDC4' for d in differences]
        bars = ax.barh(feature_names, differences, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Normalized Difference Between Clusters', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Features Distinguishing Cluster 0 vs Cluster 1', 
                    fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, differences)):
            ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.2f}', 
                   va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cluster_feature_distributions(feature_df: pd.DataFrame, 
                                          obj_features: pd.DataFrame,
                                          labels: np.ndarray,
                                          top_features: List[str],
                                          figsize: Tuple = (16, 10)):
        n_features = len(top_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            
            values_0 = obj_features[feature].values[labels == 0]
            values_1 = obj_features[feature].values[labels == 1]
            
            ax.hist(values_0, bins=30, alpha=0.6, label='Cluster 0', color='steelblue', edgecolor='black')
            ax.hist(values_1, bins=30, alpha=0.6, label='Cluster 1', color='coral', edgecolor='black')
            
            ax.set_title(feature.replace('_', ' ').title()[:25], fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions: Cluster 0 vs Cluster 1', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_clustering_comparison(results: Dict, figsize: Tuple = (15, 5)):
        k_values = list(results.keys())
        silhouette = [results[k]['silhouette'] for k in k_values]
        davies_bouldin = [results[k]['davies_bouldin'] for k in k_values]
        calinski_harabasz = [results[k]['calinski_harabasz'] for k in k_values]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].bar(k_values, silhouette, color=['#FF6B6B' if k == 2 else '#95E1D3' for k in k_values],
                   edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
        axes[0].set_title('Silhouette (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(silhouette):
            axes[0].text(k_values[i], v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        axes[1].bar(k_values, davies_bouldin, color=['#FF6B6B' if k == 2 else '#95E1D3' for k in k_values],
                   edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Davies-Bouldin Index', fontsize=11, fontweight='bold')
        axes[1].set_title('Davies-Bouldin (Lower is Better)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(davies_bouldin):
            axes[1].text(k_values[i], v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        axes[2].bar(k_values, calinski_harabasz, color=['#FF6B6B' if k == 2 else '#95E1D3' for k in k_values],
                   edgecolor='black', linewidth=2)
        axes[2].set_ylabel('Calinski-Harabasz Index', fontsize=11, fontweight='bold')
        axes[2].set_title('Calinski-Harabasz (Higher is Better)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(calinski_harabasz):
            axes[2].text(k_values[i], v + 2000, f'{int(v):,}', ha='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        return fig


def main():
    print("=" * 100)
    print("COMPREHENSIVE UNSUPERVISED CLUSTERING ANALYSIS")
    print("=" * 100)
    
    data_dir = (
        '/Users/jessicafan/Downloads/'
        'tvm board - Nov 22nd 2025 (5060 images)/'
        'TVM JSON'
    )
    output_dir = Path('./comprehensive_clustering_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("\n[1/5] Loading data and extracting features...")
    extractor = AestheticFeatureExtractor(data_dir)
    feature_df = extractor.load_all_files()
    
    obj_analyzer = ObjectiveFeatureAnalyzer(feature_df)
    scaled_features = obj_analyzer.extract_objective_features()
    print(f"Loaded {len(feature_df)} samples with {scaled_features.shape[1]} objective features")
    
    # ========================================================================
    # TEST 1: What distinguishes Cluster 0 vs Cluster 1?
    # ========================================================================
    print("\n[2/5] TEST 1: Analyzing feature importance (Cluster 0 vs Cluster 1)...")
    labels_k2, metrics_k2 = ClusterAnalyzer.cluster_data(scaled_features, k=2)
    
    importance = obj_analyzer.get_feature_importance(labels_k2)
    sorted_importance = sorted(importance.items(), 
                               key=lambda x: x[1]['difference'], 
                               reverse=True)
    
    print(f"\nTop 10 distinguishing features:")
    for i, (feature, stats) in enumerate(sorted_importance[:10], 1):
        print(f"  {i}. {feature}")
        print(f"     Cluster 0 mean: {stats['cluster_0_mean']:.4f}")
        print(f"     Cluster 1 mean: {stats['cluster_1_mean']:.4f}")
        print(f"     Difference (normalized): {stats['difference']:.4f}")
    
    fig1 = Visualizer.plot_feature_importance(importance, top_n=20)
    fig1.savefig(output_dir / 'feature_importance_k2.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("\n  ✓ Feature importance plot saved")
    
    # Distribution plots for top 12 features
    top_12_features = [f[0] for f in sorted_importance[:12]]
    fig2 = Visualizer.plot_cluster_feature_distributions(
        feature_df, obj_analyzer.objective_features, labels_k2, top_12_features
    )
    fig2.savefig(output_dir / 'feature_distributions_k2.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("  ✓ Feature distributions plot saved")
    
    
    # ========================================================================
    # TEST 2: Does k=5 emerge when removing Cluster 1?
    # ========================================================================
    print("\n[5/5] TEST 4: Testing k=5 on Cluster 0 only (removing outliers)...")
    
    # Filter to Cluster 0 only
    cluster_0_mask = labels_k2 == 0
    scaled_features_c0 = scaled_features[cluster_0_mask]
    feature_df_c0 = feature_df[cluster_0_mask].copy()
    
    print(f"Cluster 0 size: {len(feature_df_c0)} samples")
    print(f"Testing k=2, k=3, k=4, k=5, k=6 on Cluster 0 only...")
    
    clustering_results = {}
    for k in [2, 3, 4, 5, 6]:
        labels_k, metrics = ClusterAnalyzer.cluster_data(scaled_features_c0, k=k)
        clustering_results[k] = metrics
        print(f"  k={k}: Silhouette={metrics['silhouette']:.4f}, DB={metrics['davies_bouldin']:.4f}")
    
    # Find optimal k for Cluster 0
    optimal_k_c0 = max(clustering_results.keys(), 
                       key=lambda x: clustering_results[x]['silhouette'])
    print(f"\nOptimal k for Cluster 0: {optimal_k_c0} (Silhouette={clustering_results[optimal_k_c0]['silhouette']:.4f})")
    
    # If k=5 is optimal or close, analyze composition
    if optimal_k_c0 in [4, 5, 6] or clustering_results[5]['silhouette'] > 0.95:
        print("\nTesting k=5 on Cluster 0...")
        labels_k5_c0, _ = ClusterAnalyzer.cluster_data(scaled_features_c0, k=5)
        composition_k5_c0 = ClusterAnalyzer.analyze_cluster_composition(feature_df_c0, labels_k5_c0)
        
        print("\nCluster composition for k=5 (Cluster 0 only):")
        for cluster_id in sorted(composition_k5_c0.keys()):
            print(f"\n  Cluster {cluster_id} (n={composition_k5_c0[cluster_id]['size']}):")
            for modifier, count in sorted(composition_k5_c0[cluster_id]['modifiers'].items(), 
                                         key=lambda x: x[1], reverse=True):
                pct = 100 * count / composition_k5_c0[cluster_id]['size']
                print(f"    {modifier}: {count} ({pct:.1f}%)")
    else:
        print(f"\nk=5 does NOT emerge as optimal for Cluster 0 alone.")
        print(f"Optimal k is {optimal_k_c0}, indicating that even without outliers,")
        print(f"the 9 modifiers do not form 5 natural aesthetic sentiment groups.")
    
    # Create comparison plot
    fig3 = Visualizer.plot_clustering_comparison(clustering_results)
    fig3.savefig(output_dir / 'clustering_comparison_cluster0_only.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("\n  ✓ Clustering comparison (Cluster 0 only) saved")
    
    # Save comprehensive results - convert numpy types
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
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
    
    results_summary = {
        'total_samples': len(feature_df),
        'cluster_0_size': int(composition[0]['size']),
        'cluster_1_size': int(composition[1]['size']),
        'k2_metrics': {k: float(v) for k, v in metrics_k2.items()},
        'top_10_distinguishing_features': [
            {
                'feature': f[0],
                'difference': float(f[1]['difference']),
                'cluster_0_mean': float(f[1]['cluster_0_mean']),
                'cluster_1_mean': float(f[1]['cluster_1_mean'])
            } for f in sorted_importance[:10]
        ],
        'cluster_0_modifiers': {str(k): int(v) for k, v in composition[0]['modifiers'].items()},
        'cluster_1_modifiers': {str(k): int(v) for k, v in composition[1]['modifiers'].items()},
        'cluster_0_only_analysis': {
            'optimal_k': int(optimal_k_c0),
            'silhouette_scores': {str(k): float(v['silhouette']) for k, v in clustering_results.items()},
            'davies_bouldin_scores': {str(k): float(v['davies_bouldin']) for k, v in clustering_results.items()}
        }
    }
    
    results_summary = convert_to_serializable(results_summary)
    
    with open(output_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
if __name__ == "__main__":
    main()
