import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.stats import spearmanr
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PCoAMovieRecommendationSystem:
    """
    Complete PCoA pipeline using Weighted Manhattan Distance & IDF
    """
    
    def __init__(self, data_path='Cleaned_data.csv'):
        print("="*80)
        print("WEIGHTED MANHATTAN PCoA MOVIE RECOMMENDATION SYSTEM")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("Loading movie data...")
        self.data = pd.read_csv(data_path)
        
        # Detect ID column
        id_candidates = [col for col in self.data.columns if 'imdb' in col.lower() or 
                         col.lower() in ['id', 'movie_id', 'title_id']]
        
        if id_candidates:
            self.id_col = id_candidates[0]
            self.movie_ids = self.data[self.id_col].values
            print(f"✓ ID column: '{self.id_col}'")
        else:
            self.movie_ids = self.data.index.values
            self.id_col = None
            print("⚠ Using index as ID")
        
        # Get numeric traits (features)
        self.feature_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.id_col and self.id_col in self.feature_cols:
            self.feature_cols.remove(self.id_col)
            
        print(f"✓ Loaded: {len(self.data):,} movies")
        print(f"✓ Features: {len(self.feature_cols)}")
        
        self.pcoa_coords = None
        self.correlations = None
        self.eigen_spectrum = None

    def analyze_eigen_spectrum(self, eigenvalues, output_prefix='pcoa'):
        """Analyze eigen spectrum from decomposition and save scree/variance outputs."""
        positive_eigs = np.maximum(eigenvalues, 0)
        total_positive = positive_eigs.sum()

        if total_positive > 0:
            explained_ratio = positive_eigs / total_positive
        else:
            explained_ratio = np.zeros_like(positive_eigs)

        cumulative_ratio = np.cumsum(explained_ratio)

        # Elbow estimate via max second derivative of log-scaled positive eigenvalues
        safe_eigs = np.where(positive_eigs > 0, positive_eigs, 1e-12)
        log_eigs = np.log(safe_eigs)
        if len(log_eigs) >= 3:
            curvature = np.diff(log_eigs, n=2)
            elbow_dim = int(np.argmax(np.abs(curvature)) + 2)
        else:
            elbow_dim = 1

        spectrum_df = pd.DataFrame({
            'dimension': np.arange(1, len(eigenvalues) + 1),
            'eigenvalue': eigenvalues,
            'positive_eigenvalue': positive_eigs,
            'explained_variance_ratio': explained_ratio,
            'cumulative_explained_variance': cumulative_ratio
        })

        spectrum_csv = f'{output_prefix}_eigen_spectrum.csv'
        spectrum_df.to_csv(spectrum_csv, index=False)
        self.eigen_spectrum = spectrum_df

        print(f"  ✓ Saved eigen spectrum table: {spectrum_csv}")
        print(f"  ✓ Estimated elbow at Dimension {elbow_dim}")

        if len(cumulative_ratio) >= 20:
            print(f"  ✓ Cumulative variance by 10 dims: {cumulative_ratio[9] * 100:.2f}%")
            print(f"  ✓ Cumulative variance by 20 dims: {cumulative_ratio[19] * 100:.2f}%")

        # Scree plot visualization
        try:
            import matplotlib.pyplot as plt

            max_dims_to_plot = min(30, len(positive_eigs))
            x = np.arange(1, max_dims_to_plot + 1)
            y = positive_eigs[:max_dims_to_plot]

            plt.figure(figsize=(10, 6))
            plt.plot(x, y, marker='o', linewidth=2, markersize=5, label='Eigenvalue')
            plt.axvline(elbow_dim, color='red', linestyle='--', linewidth=1.5,
                        label=f'Elbow ≈ Dim {elbow_dim}')
            plt.title('PCoA Scree Plot (Eigenvalue Decomposition)')
            plt.xlabel('Dimension')
            plt.ylabel('Eigenvalue (non-negative for variance)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            scree_png = f'{output_prefix}_scree_plot.png'
            plt.savefig(scree_png, dpi=200)
            plt.close()
            print(f"  ✓ Saved scree plot: {scree_png}")
        except Exception as plot_err:
            print(f"  ⚠ Could not generate scree plot: {plot_err}")

        return elbow_dim, spectrum_df

    def step1_run_weighted_pcoa(self, n_components=20):
        """STEP 1: Calculate true Classical MDS using IDF Weighted Manhattan"""
        print("\n" + "="*80)
        print(f"STEP 1: WEIGHTED MANHATTAN PCoA ({n_components}D)")
        print("="*80)
        
        trait_matrix = self.data[self.feature_cols].fillna(0).values
        
        # 1. IDF Weights (Tempered)
        print("\n[1/4] Calculating IDF Weights...")
        N = trait_matrix.shape[0]
        df_counts = np.sum(trait_matrix > 0, axis=0)
        base_weights = np.log((N + 1) / (df_counts + 1)) + 1
        weights = base_weights ** 0.85
        print(f"  ✓ Upweight factor: {weights.max() / weights.min():.1f}x")
        
        # 2. Weighted Manhattan
        print("\n[2/4] Computing Weighted Manhattan Distances...")
        weighted_traits = trait_matrix * weights
        distance_matrix = pairwise_distances(weighted_traits, metric='manhattan', n_jobs=-1)
        
        # Normalize
        nonzero = distance_matrix[distance_matrix > 0]
        dmin, dmax = nonzero.min(), distance_matrix.max()
        normalized = (distance_matrix - dmin) / (dmax - dmin)
        np.fill_diagonal(normalized, 0)
        
        # 3. Eigenvalue decomposition
        print("\n[3/4] Running eigenvalue decomposition...")
        n = normalized.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (normalized ** 2) @ H
        
        eigenvalues, eigenvectors = eigh(B)
        
        # Sort descending
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        print("\n[3b/4] Eigen spectrum analysis (elbow + scree)...")
        self.analyze_eigen_spectrum(eigenvalues, output_prefix='pcoa')
        
        # 4. Extract Coordinates
        print(f"\n[4/4] Extracting {n_components} coordinates...")
        top_eigenvalues = eigenvalues[:n_components]
        top_eigenvectors = eigenvectors[:, :n_components]
        
        coords = top_eigenvectors @ np.diag(np.sqrt(np.abs(top_eigenvalues)))
        
        # Save results
        dim_cols = [f'PCoA_Dim{i+1}' for i in range(n_components)]
        self.pcoa_coords = pd.DataFrame(coords, columns=dim_cols)
        
        if self.id_col:
            self.pcoa_coords.insert(0, self.id_col, self.movie_ids)
            
        output_file = f'pcoa_coordinates_{n_components}D.csv'
        self.pcoa_coords.to_csv(output_file, index=False)
        
        print(f"\n✓ STEP 1 COMPLETE: Saved to {output_file}")
        return self.pcoa_coords

    def step2_calculate_correlations(self):
        """STEP 2: Calculate Feature-Dimension Correlations (Spearman)"""
        print("\n" + "="*80)
        print("STEP 2: FEATURE-DIMENSION CORRELATION ANALYSIS")
        print("="*80)
        
        # Merge coords with original data
        if self.id_col:
            full_data = pd.merge(self.data, self.pcoa_coords, on=self.id_col, how='inner')
        else:
            full_data = pd.concat([self.data, self.pcoa_coords], axis=1)
            
        dim_cols = [col for col in self.pcoa_coords.columns if col.startswith('PCoA_Dim')]
        
        print(f"\nCalculating Spearman correlations for {len(dim_cols)} dimensions...")
        all_correlations = []
        
        for dim in dim_cols:
            for feature in self.feature_cols:
                x = full_data[dim].values
                y = full_data[feature].values
                
                # Failsafe for zero-variance
                if len(np.unique(y)) <= 1:
                    continue 
                    
                corr_coef, p_value = spearmanr(x, y)
                
                if pd.isna(corr_coef):
                    corr_coef, p_value = 0.0, 1.0
                    
                all_correlations.append({
                    'dimension': dim,
                    'feature': feature,
                    'correlation': corr_coef
                })
                
        self.correlations = pd.DataFrame(all_correlations)
        self.correlations.to_csv('dimension_feature_correlations.csv', index=False)
        print("✓ STEP 2 COMPLETE: Saved to dimension_feature_correlations.csv")

    def step3_interpret_dimensions(self, top_n=15):
        """STEP 3: Generate the Vibe Report"""
        print("\n" + "="*80)
        print("STEP 3: GENERATING VIBE REPORT")
        print("="*80)
        
        dim_cols = [col for col in self.pcoa_coords.columns if col.startswith('PCoA_Dim')]
        report_lines = ["NEW WEIGHTED MANHATTAN VIBE REPORT\n" + "="*40]
        
        for dim_name in dim_cols:
            dim_data = self.correlations[self.correlations['dimension'] == dim_name].copy()
            
            # Show top Positive Traits
            top_traits = dim_data.nlargest(top_n, 'correlation')
            report_lines.append(f"\n[{dim_name}] POSITIVE TRAITS:")
            for _, row in top_traits.iterrows():
                if row['correlation'] > 0.1: 
                    report_lines.append(f"  +{row['correlation']:.3f} | {row['feature']}")
            
            # Show top Negative Traits (just in case Manhattan creates pure opposites)
            bottom_traits = dim_data.nsmallest(5, 'correlation')
            has_negatives = False
            for _, row in bottom_traits.iterrows():
                if row['correlation'] < -0.1:
                    if not has_negatives:
                        report_lines.append(f"  --- Negative Traits ---")
                        has_negatives = True
                    report_lines.append(f"  {row['correlation']:.3f} | {row['feature']}")
            report_lines.append("\n")
            
        report_text = "\n".join(report_lines)
        
        with open('vibe_dimensions_report.txt', 'w') as f:
            f.write(report_text)
            
        print("✓ STEP 3 COMPLETE: Open 'vibe_dimensions_report.txt' to see the new dimensions!")

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    recommender = PCoAMovieRecommendationSystem('Cleaned_data.csv')
    
    # Run the new Weighted Manhattan 20D pipeline
    recommender.step1_run_weighted_pcoa(n_components=20)
    recommender.step2_calculate_correlations()
    recommender.step3_interpret_dimensions()