import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class FeatureExtractor:
    
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
    
    def __init__(self, data_dir: str, dataset_name: str = "Unknown"):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.raw_features = []
        self.metadata = []
        self.labels = []
        self.feature_df = None
        self.scaler = RobustScaler()
        self.scaled_features = None
        
    def load_all_files(self):
        json_files = sorted(list(self.data_dir.glob('*.json')))
        print(f"[{self.dataset_name}] Found {len(json_files)} JSON files")
        
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
                print(f"  Error loading {json_file}: {e}")
        
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
    
    def extract_features(self):
        candidate_cols = self._get_all_feature_names()
        available_cols = [col for col in candidate_cols if col in self.feature_df.columns]
        
        objective_features = self.feature_df[available_cols].fillna(0)
        self.scaled_features = self.scaler.fit_transform(objective_features)
        
        print(f"[{self.dataset_name}] Extracted {self.scaled_features.shape[1]} features from {len(self.feature_df)} samples")
        return self.scaled_features
    
    def _get_all_feature_names(self):
        all_features = []
        for category_name, category_features in self.FEATURE_CATEGORIES.items():
            for f in category_features:
                all_features.append(f"{category_name}_{f}")
        return all_features


def analyze_dataset(extractor: FeatureExtractor, k_range: range = range(2, 12)) -> Dict:
    print(f"\n[{extractor.dataset_name}] Testing k={min(k_range)} to {max(k_range)}...")
    
    results = {
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'k_range': list(k_range)
    }
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(extractor.scaled_features)
        
        sil = silhouette_score(extractor.scaled_features, labels)
        db = davies_bouldin_score(extractor.scaled_features, labels)
        ch = calinski_harabasz_score(extractor.scaled_features, labels)
        
        results['silhouette'].append(sil)
        results['davies_bouldin'].append(db)
        results['calinski_harabasz'].append(ch)
        
        print(f"  k={k}: Silhouette={sil:.4f}, DB={db:.4f}, CH={ch:.0f}")

    optimal_k_sil = k_range[np.argmax(results['silhouette'])]
    results['optimal_k'] = optimal_k_sil
    results['optimal_silhouette'] = max(results['silhouette'])
    
    return results


def plot_comparison(curated_results: Dict, benchmark_results: Dict, figsize: Tuple = (16, 5)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    k_range_you = curated_results['k_range']
    k_range_bench = benchmark_results['k_range']
    
    axes[0].plot(k_range_you, curated_results['silhouette'], 'o-', 
                label='Curated Data (9 modifiers)', linewidth=2.5, markersize=8, color='#FF6B6B')
    axes[0].plot(k_range_bench, benchmark_results['silhouette'], 's-',
                label='Benchmark Data (8 emotions)', linewidth=2.5, markersize=8, color='#4ECDC4')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Silhouette Score (Higher is Better)', fontsize=12, fontweight='bold')
    axes[0].set_title('Silhouette Comparison', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range_you, curated_results['davies_bouldin'], 'o-',
                label='Curated_Data', linewidth=2.5, markersize=8, color='#FF6B6B')
    axes[1].plot(k_range_bench, benchmark_results['davies_bouldin'], 's-',
                label='Benchmark Data', linewidth=2.5, markersize=8, color='#4ECDC4')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].set_title('Davies-Bouldin Comparison', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(k_range_you, curated_results['calinski_harabasz'], 'o-',
                label='Your Data', linewidth=2.5, markersize=8, color='#FF6B6B')
    axes[2].plot(k_range_bench, benchmark_results['calinski_harabasz'], 's-',
                label='Benchmark Data', linewidth=2.5, markersize=8, color='#4ECDC4')
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Calinski-Harabasz Index (Higher is Better)', fontsize=12, fontweight='bold')
    axes[2].set_title('Calinski-Harabasz Comparison', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    print("\n" + "=" * 100)
    print("CROSS-DATASET ANALYSIS: Curated Data vs Benchmark Data")
    print("=" * 100)
    
    output_dir = Path('./cross_dataset_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # YOUR DATA
    print("\n[LOADING COLLECTED DATA]")
    your_data_dir = Path(
        '/Users/jessicafan/Downloads/'
        'tvm board - Nov 22nd 2025 (5060 images)/'
        'TVM JSON'
    )
    
    your_extractor = FeatureExtractor(your_data_dir, "Curated Data")
    your_df = your_extractor.load_all_files()
    your_scaled = your_extractor.extract_features()
    your_results = analyze_dataset(your_extractor, k_range=range(2, 12))
    
    print("\n[LOADING BENCHMARK DATA]")
    benchmark_data_dir = Path(
        '/Users/jessicafan/Library/CloudStorage/GoogleDrive-tvm.tiervibemap@gmail.com/'
        'My Drive/tvm/per_file_results'
    )
    
    if benchmark_data_dir.exists():
        subdirs = [d for d in benchmark_data_dir.iterdir() if d.is_dir()]
        print(f"Found {len(subdirs)} emotion categories:")
        for d in sorted(subdirs):
            json_count = len(list(d.glob('*.json')))
            print(f"  - {d.name}: {json_count} images")
        
        all_benchmark_features = []
        all_benchmark_metadata = []
        all_benchmark_labels = []
        
        for emotion_dir in sorted(subdirs):
            json_files = sorted(list(emotion_dir.glob('*.json')))
            print(f"\nLoading {emotion_dir.name}... ({len(json_files)} files)")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    all_benchmark_features.append(data['features'])
                    all_benchmark_metadata.append(data['meta'])
                    
                    filename_parts = json_file.stem.split('_')
                    emotion = filename_parts[0]  
                    all_benchmark_labels.append(emotion)
                except Exception as e:
                    print(f"  Error: {e}")
        
        print(f"\nTotal benchmark samples loaded: {len(all_benchmark_features)}")
        
        from collections import Counter
        emotion_counts = Counter(all_benchmark_labels)
        print(f"Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count}")
        
        bench_extractor = FeatureExtractor(benchmark_data_dir, "Benchmark Data")
        bench_extractor.raw_features = all_benchmark_features
        bench_extractor.metadata = all_benchmark_metadata
        bench_extractor.labels = all_benchmark_labels
        bench_extractor.feature_df = bench_extractor._flatten_to_dataframe()
        bench_scaled = bench_extractor.extract_features()
        bench_results = analyze_dataset(bench_extractor, k_range=range(2, 12))
        
        print("\n" + "=" * 100)
        print("CROSS-DATASET COMPARISON")
        print("=" * 100)
        
        print(f"\nCurated Data:")
        print(f"  Samples: {len(your_df)}")
        print(f"  Categories: {your_df['base_modifier'].nunique()} modifiers")
        print(f"  Optimal k: {your_results['optimal_k']} (Silhouette: {your_results['optimal_silhouette']:.4f})")
        
        print(f"\nBenchmark Data:")
        print(f"  Samples: {len(bench_extractor.feature_df)}")
        print(f"  Categories: {bench_extractor.feature_df['base_modifier'].nunique()} emotions")
        print(f"  Optimal k: {bench_results['optimal_k']} (Silhouette: {bench_results['optimal_silhouette']:.4f})")
        
        fig = plot_comparison(your_results, bench_results)
        fig.savefig(output_dir / 'cross_dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n✓ Comparison plot saved: cross_dataset_comparison.png")

        summary = {
            'curated_data': {
                'samples': int(len(your_df)),
                'categories': int(your_df['base_modifier'].nunique()),
                'optimal_k': int(your_results['optimal_k']),
                'optimal_silhouette': float(your_results['optimal_silhouette']),
                'silhouette_by_k': {str(k): float(s) for k, s in zip(your_results['k_range'], your_results['silhouette'])}
            },
            'benchmark_data': {
                'samples': int(len(bench_extractor.feature_df)),
                'categories': int(bench_extractor.feature_df['base_modifier'].nunique()),
                'optimal_k': int(bench_results['optimal_k']),
                'optimal_silhouette': float(bench_results['optimal_silhouette']),
                'silhouette_by_k': {str(k): float(s) for k, s in zip(bench_results['k_range'], bench_results['silhouette'])}
            },
            'interpretation': {
                'same_optimal_k': your_results['optimal_k'] == bench_results['optimal_k'],
                'k5_natural_in_your_data': abs(your_k5_sil - your_results['optimal_silhouette']) < 0.01,
                'k8_natural_in_benchmark': bench_results['optimal_k'] == 8
            }
        }
        
        with open(output_dir / 'cross_dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved: cross_dataset_summary.json")
        print(f"\nAll results saved to: {output_dir}")
        
    else:
        print(f"ERROR: Benchmark data directory not found: {benchmark_data_dir}")
        print("Please verify the path.")


if __name__ == "__main__":
    main()
