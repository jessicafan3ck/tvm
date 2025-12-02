import subprocess
import sys

print("=" * 80)
print("Installing CLIP and dependencies...")
print("=" * 80)

# Install required packages
packages = [
    'openai-clip',
    'torch',
    'torchvision',
]

for package in packages:
    print(f"\nInstalling {package}...")
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', package, '--break-system-packages'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✓ {package} installed successfully")
    except Exception as e:
        print(f"⚠ {package} installation note: {e}")

print("\n" + "=" * 80)
print("CLIP setup complete. Now running enhanced analysis...")
print("=" * 80 + "\n")

# Now import and run the analysis
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

# Try to import CLIP
try:
    import clip
    import torch
    HAS_CLIP = True
    print("✓ CLIP imported successfully\n")
except ImportError as e:
    HAS_CLIP = False
    print(f"⚠ CLIP import failed: {e}\n")


class CLIPSemanticExtractor:
    """Extract semantic embeddings using CLIP."""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        if not HAS_CLIP:
            raise ImportError("CLIP not available. Run: pip install openai-clip torch")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.embedding_cache = {}
        print(f"✓ CLIP loaded\n")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate CLIP embedding for text."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        with torch.no_grad():
            text_tokens = clip.tokenize(text).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)
            embedding = text_embedding.cpu().numpy()[0]
            self.embedding_cache[text] = embedding
            return embedding
    
    def embed_modifiers(self, modifiers: List[str]) -> np.ndarray:
        """Generate embeddings for all modifiers."""
        print(f"Generating CLIP embeddings for {len(modifiers)} modifiers...")
        embeddings = []
        for i, modifier in enumerate(modifiers):
            embedding = self.embed_text(modifier)
            embeddings.append(embedding)
            if (i + 1) % 3 == 0:
                print(f"  {i + 1}/{len(modifiers)} embeddings generated")
        
        print(f"✓ Generated {len(embeddings)} CLIP embeddings (512-dim)\n")
        return np.array(embeddings)


class AestheticFeatureExtractor:
    """Extract objective features from JSONs."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.raw_features = []
        self.metadata = []
        self.labels = []
        self.feature_df = None
        
    def load_all_files(self) -> pd.DataFrame:
        """Load all JSON files."""
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
    
    def _flatten_to_dataframe(self) -> pd.DataFrame:
        """Flatten nested features to dataframe."""
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
        print(f"Extracted {len(self.feature_df)} samples")
        print(f"Unique base modifiers: {len(self.feature_df['base_modifier'].unique())}\n")
        return self.feature_df


class ObjectiveFeatureAnalyzer:
    """Extract objective visual features."""
    
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
        """Extract F_obj."""
        candidate_cols = self._get_all_feature_names()
        available_cols = [col for col in candidate_cols if col in self.feature_df.columns]

        if not available_cols:
            raise ValueError("No objective features found.")
        
        self.objective_features = self.feature_df[available_cols].fillna(0)
        self.scaled_features = self.scaler.fit_transform(self.objective_features)
        
        print(f"Extracted {self.objective_features.shape[1]} objective features (F_obj)\n")
        return self.scaled_features
    
    def _get_all_feature_names(self) -> List[str]:
        """Get all feature names."""
        all_features = []
        for category_name, category_features in self.FEATURE_CATEGORIES.items():
            for f in category_features:
                all_features.append(f"{category_name}_{f}")
        return all_features


class ModifierDisentanglement:
    """Aggregate features by modifier (disentangle from subjects)."""
    
    def __init__(self, feature_df: pd.DataFrame, scaled_objective_features: np.ndarray):
        self.feature_df = feature_df
        self.scaled_objective_features = scaled_objective_features
        self.modifier_aggregates = {}
        self.modifier_features = None
        
    def aggregate_by_modifier(self) -> Tuple[np.ndarray, List[str]]:
        """Average objective features across subjects for each modifier."""
        modifiers = sorted(self.feature_df['base_modifier'].unique())
        aggregated = []
        
        for modifier in modifiers:
            mask = self.feature_df['base_modifier'] == modifier
            modifier_features = self.scaled_objective_features[mask]
            mean_features = modifier_features.mean(axis=0)
            aggregated.append(mean_features)
            self.modifier_aggregates[modifier] = {
                'mean': mean_features,
                'std': modifier_features.std(axis=0),
                'n_samples': mask.sum()
            }
        
        self.modifier_features = np.array(aggregated)
        print(f"Aggregated {len(modifiers)} base modifiers (F_obj averaged across subjects)\n")
        return self.modifier_features, modifiers


class SentimentClusterer:
    """Cluster modifiers by sentiment using different feature combinations."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.results = {}
        
    def cluster_on_features(self, features: np.ndarray, modifiers: List[str], 
                           feature_name: str = "features",
                           k_range: range = range(2, 10)) -> Dict:
        """Cluster modifiers on given features."""
        
        # Adjust k range for sample size
        n_modifiers = features.shape[0]
        k_range = range(2, min(n_modifiers, max(k_range) + 1))
        
        scaled = self.scaler.fit_transform(features)
        
        metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled)
            
            metrics['silhouette'].append(silhouette_score(scaled, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(scaled, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(scaled, labels))
        
        optimal_k = k_range[np.argmax(metrics['silhouette'])]
        
        # Get final clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans_final.fit_predict(scaled)
        
        # Group modifiers
        groups = {}
        for modifier, label in zip(modifiers, labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(modifier)
        
        return {
            'feature_name': feature_name,
            'n_features': features.shape[1],
            'optimal_k': optimal_k,
            'silhouette': float(metrics['silhouette'][optimal_k - min(k_range)]),
            'davies_bouldin': float(metrics['davies_bouldin'][optimal_k - min(k_range)]),
            'calinski_harabasz': float(metrics['calinski_harabasz'][optimal_k - min(k_range)]),
            'labels': labels,
            'groups': groups,
            'silhouette_scores': metrics['silhouette'],
            'k_range': list(k_range)
        }


class Visualizer:
    """Create visualizations comparing clustering approaches."""
    
    @staticmethod
    def plot_clustering_comparison(clustering_results: List[Dict], figsize: Tuple = (16, 5)):
        """Compare clustering results across different feature spaces."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for idx, result in enumerate(clustering_results):
            k_range = result['k_range']
            silhouette = result['silhouette_scores']
            
            axes[idx].plot(k_range, silhouette, 'o-', linewidth=2.5, markersize=8, color='#FF6B6B')
            axes[idx].axvline(result['optimal_k'], color='green', linestyle='--', linewidth=2.5, label=f'Optimal k={result["optimal_k"]}')
            axes[idx].set_xlabel('Number of Groups (k)', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{result["feature_name"]}\n({result["n_features"]} features)', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(fontsize=10)
            axes[idx].set_ylim([min(silhouette) - 0.05, max(silhouette) + 0.05])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_modifier_groups(clustering_results: Dict, title: str = "Semantic Sentiment Groups"):
        """Visualize modifier groupings."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        groups = clustering_results['groups']
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        y_pos = 0
        for (group_id, modifiers), color in zip(sorted(groups.items()), colors):
            group_text = f"Group {group_id}: {', '.join(modifiers)}"
            ax.text(0.05, y_pos, group_text, fontsize=12, color='black', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.8, pad=0.8))
            y_pos -= 0.12
        
        ax.set_xlim(0, 1)
        ax.set_ylim(y_pos - 0.1, 0.15)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_2d_modifier_space(reduced_features: np.ndarray, modifiers: List[str],
                               labels: np.ndarray, title: str = "Modifier Space"):
        """Plot modifiers in 2D reduced space with group coloring."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(
            reduced_features[:, 0], reduced_features[:, 1],
            c=labels, cmap='tab10', s=400, alpha=0.7, edgecolors='black', linewidth=2
        )
        
        for i, modifier in enumerate(modifiers):
            ax.annotate(
                modifier, 
                (reduced_features[i, 0], reduced_features[i, 1]),
                fontsize=11, fontweight='bold', ha='center', va='center'
            )
        
        plt.colorbar(scatter, ax=ax, label='Sentiment Group')
        ax.set_xlabel('Component 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Component 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig


def main():
    """Run the enhanced analysis."""
    
    print("=" * 80)
    print("ENHANCED AESTHETIC LATENT SPACE ANALYSIS")
    print("Comparing F_obj, F_sem, and combined approaches")
    print("=" * 80 + "\n")
    
    # Configuration
    data_dir = (
        '/Users/jessicafan/Downloads/'
        'tvm board - Nov 22nd 2025 (5060 images)/'
        'TVM JSON'
    )
    output_dir = Path('./aesthetic_analysis_with_clip_results')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Extract features
    print("[1/7] Extracting Features...")
    extractor = AestheticFeatureExtractor(data_dir)
    feature_df = extractor.load_all_files()
    
    # 2. Extract objective features (F_obj)
    print("[2/7] Extracting Objective Features (F_obj)...")
    obj_analyzer = ObjectiveFeatureAnalyzer(feature_df)
    scaled_obj_features = obj_analyzer.extract_objective_features()
    
    # 3. Disentangle modifiers
    print("[3/7] Disentangling Modifiers from Subjects...")
    disentanglement = ModifierDisentanglement(feature_df, scaled_obj_features)
    modifier_obj_features, modifiers = disentanglement.aggregate_by_modifier()
    
    print(f"Modifiers: {modifiers}\n")
    
    # 4. Extract semantic features (F_sem) via CLIP
    print("[4/7] Extracting Semantic Embeddings (F_sem via CLIP)...")
    modifier_sem_features = None
    if HAS_CLIP:
        try:
            clip_extractor = CLIPSemanticExtractor()
            modifier_sem_features = clip_extractor.embed_modifiers(modifiers)
        except Exception as e:
            print(f"⚠ CLIP extraction failed: {e}\n")
    else:
        print("⚠ CLIP not available - will skip semantic clustering\n")
    
    # 5. Reduce modifier space for visualization
    print("[5/7] Reducing Modifier Latent Space...")
    reducer_pca = PCA(n_components=2)
    modifier_pca = reducer_pca.fit_transform(modifier_obj_features)
    print(f"PCA: Reduced to 2 dimensions (explains {reducer_pca.explained_variance_ratio_.sum():.2%} variance)\n")
    
    if modifier_sem_features is not None:
        reducer_pca_sem = PCA(n_components=2)
        modifier_pca_sem = reducer_pca_sem.fit_transform(modifier_sem_features)
        print(f"PCA (semantic): Reduced to 2 dimensions (explains {reducer_pca_sem.explained_variance_ratio_.sum():.2%} variance)\n")
    else:
        modifier_pca_sem = None
    
    # 6. Cluster on different feature spaces
    print("[6/7] Clustering Modifiers by Sentiment...")
    clusterer = SentimentClusterer()
    
    results = {}
    
    # Objective features only
    print("  Testing F_obj (objective features only)...")
    results['f_obj'] = clusterer.cluster_on_features(
        modifier_obj_features, modifiers, 
        feature_name='F_obj (Objective Features)',
        k_range=range(2, 10)
    )
    print(f"    → Optimal k: {results['f_obj']['optimal_k']}, Silhouette: {results['f_obj']['silhouette']:.4f}")
    
    # Semantic features only
    if modifier_sem_features is not None:
        print("  Testing F_sem (semantic embeddings only)...")
        results['f_sem'] = clusterer.cluster_on_features(
            modifier_sem_features, modifiers,
            feature_name='F_sem (CLIP Semantic Embeddings)',
            k_range=range(2, 10)
        )
        print(f"    → Optimal k: {results['f_sem']['optimal_k']}, Silhouette: {results['f_sem']['silhouette']:.4f}")
        
        # Combined features
        print("  Testing F_obj + F_sem (combined)...")
        # Normalize both before combining
        scaler_obj = StandardScaler()
        scaler_sem = StandardScaler()
        norm_obj = scaler_obj.fit_transform(modifier_obj_features)
        norm_sem = scaler_sem.fit_transform(modifier_sem_features)
        combined_features = np.hstack([norm_obj, norm_sem])
        
        results['combined'] = clusterer.cluster_on_features(
            combined_features, modifiers,
            feature_name='F_obj + F_sem (Combined)',
            k_range=range(2, 10)
        )
        print(f"    → Optimal k: {results['combined']['optimal_k']}, Silhouette: {results['combined']['silhouette']:.4f}\n")
    
    # 7. Visualizations and summary
    print("[7/7] Creating Visualizations...")
    
    # Comparison plot
    clustering_list = [results['f_obj']]
    if 'f_sem' in results:
        clustering_list.append(results['f_sem'])
    if 'combined' in results:
        clustering_list.append(results['combined'])
    
    fig1 = Visualizer.plot_clustering_comparison(clustering_list, figsize=(15, 5))
    fig1.savefig(output_dir / 'clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("  ✓ Clustering comparison saved")
    
    # Objective features groups
    fig2 = Visualizer.plot_modifier_groups(results['f_obj'], "Sentiment Groups (F_obj)")
    fig2.savefig(output_dir / 'groups_f_obj.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("  ✓ F_obj groups visualization saved")
    
    # Objective features 2D
    fig3 = Visualizer.plot_2d_modifier_space(
        modifier_pca, modifiers, results['f_obj']['labels'],
        title="Modifier Latent Space (F_obj, k=2)"
    )
    fig3.savefig(output_dir / 'modifier_space_f_obj.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("  ✓ F_obj modifier space saved")
    
    # Semantic features
    if modifier_sem_features is not None:
        fig4 = Visualizer.plot_modifier_groups(results['f_sem'], "Sentiment Groups (F_sem)")
        fig4.savefig(output_dir / 'groups_f_sem.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("  ✓ F_sem groups visualization saved")
        
        fig5 = Visualizer.plot_2d_modifier_space(
            modifier_pca_sem, modifiers, results['f_sem']['labels'],
            title="Modifier Latent Space (F_sem, k=2)"
        )
        fig5.savefig(output_dir / 'modifier_space_f_sem.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print("  ✓ F_sem modifier space saved")
        
        fig6 = Visualizer.plot_modifier_groups(results['combined'], "Sentiment Groups (F_obj + F_sem)")
        fig6.savefig(output_dir / 'groups_combined.png', dpi=300, bbox_inches='tight')
        plt.close(fig6)
        print("  ✓ Combined groups visualization saved")
    
    # Save summary
    summary = {
        'modifiers': modifiers,
        'clustering_results': {
            'f_obj': {
                'optimal_k': int(results['f_obj']['optimal_k']),
                'silhouette': float(results['f_obj']['silhouette']),
                'davies_bouldin': float(results['f_obj']['davies_bouldin']),
                'calinski_harabasz': float(results['f_obj']['calinski_harabasz']),
                'groups': {str(k): v for k, v in results['f_obj']['groups'].items()}
            }
        }
    }
    
    if 'f_sem' in results:
        summary['clustering_results']['f_sem'] = {
            'optimal_k': int(results['f_sem']['optimal_k']),
            'silhouette': float(results['f_sem']['silhouette']),
            'davies_bouldin': float(results['f_sem']['davies_bouldin']),
            'calinski_harabasz': float(results['f_sem']['calinski_harabasz']),
            'groups': {str(k): v for k, v in results['f_sem']['groups'].items()}
        }
    
    if 'combined' in results:
        summary['clustering_results']['combined'] = {
            'optimal_k': int(results['combined']['optimal_k']),
            'silhouette': float(results['combined']['silhouette']),
            'davies_bouldin': float(results['combined']['davies_bouldin']),
            'calinski_harabasz': float(results['combined']['calinski_harabasz']),
            'groups': {str(k): v for k, v in results['combined']['groups'].items()}
        }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - RESULTS")
    print("=" * 80 + "\n")
    
    print("CLUSTERING RESULTS COMPARISON:\n")
    print(f"F_obj (Objective Features):")
    print(f"  Optimal k: {results['f_obj']['optimal_k']}")
    print(f"  Silhouette: {results['f_obj']['silhouette']:.4f}")
    print(f"  Groups: {results['f_obj']['groups']}\n")
    
    if 'f_sem' in results:
        print(f"F_sem (CLIP Semantic Embeddings):")
        print(f"  Optimal k: {results['f_sem']['optimal_k']}")
        print(f"  Silhouette: {results['f_sem']['silhouette']:.4f}")
        print(f"  Groups: {results['f_sem']['groups']}\n")
        
        print(f"Combined (F_obj + F_sem):")
        print(f"  Optimal k: {results['combined']['optimal_k']}")
        print(f"  Silhouette: {results['combined']['silhouette']:.4f}")
        print(f"  Groups: {results['combined']['groups']}\n")
    
    print(f"Results saved to: {output_dir}\n")
    
    print("=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("""
1. F_obj (Objective) clustering shows what visual features distinguish modifiers
2. F_sem (Semantic) clustering shows pure semantic/linguistic relationships
3. Combined approach tests if modifiers need both visual + semantic structure
4. Compare silhouette scores: higher = better separation

INTERPRETATION GUIDE:
- If F_sem k is 5: semantic structure has 5-group separation
- If F_obj k is 2: visual features only distinguish 2 groups (likely whimsical vs others)
- If combined k is 5: confirms your 5-group hypothesis
""")


if __name__ == "__main__":
    main()
