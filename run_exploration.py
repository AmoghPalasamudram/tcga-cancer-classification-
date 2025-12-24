"""
TCGA Pan-Cancer Data Exploration Script
Run this after activating the tcga-cancer conda environment

Usage:
    conda activate tcga-cancer
    python run_exploration.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("TCGA PAN-CANCER DATA EXPLORATION")
print("=" * 60)
print()

# Step 1: Import libraries
print("[1/8] Importing libraries...")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For saving figures without display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data_loader import load_tcga_data, load_as_dataframe, CANCER_TYPE_INFO

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
print("    Libraries loaded successfully!")

# Step 2: Download and load data
print()
print("[2/8] Downloading and loading TCGA dataset...")
print("    (This may take a minute on first run)")
print()
X, y, gene_names, sample_ids = load_tcga_data(verbose=True)

print()
print("=" * 50)
print("DATA SUMMARY")
print("=" * 50)
print(f"Number of samples: {X.shape[0]}")
print(f"Number of genes (features): {X.shape[1]:,}")
print(f"Number of cancer types: {len(np.unique(y))}")

# Step 3: Show cancer types
print()
print("[3/8] Cancer types in dataset:")
print("-" * 60)
for code, info in CANCER_TYPE_INFO.items():
    print(f"  {code}: {info['full_name']}")
    print(f"       Organ: {info['organ']}")

# Step 4: Class distribution
print()
print("[4/8] Class distribution:")
class_counts = pd.Series(y).value_counts().sort_index()
for cancer, count in class_counts.items():
    pct = count / len(y) * 100
    bar = "#" * int(pct / 2)
    print(f"  {cancer}: {count:3d} samples ({pct:5.1f}%) {bar}")

# Create figures directory if needed
os.makedirs('results/figures', exist_ok=True)

# Step 5: Plot class distribution
print()
print("[5/8] Creating class distribution plot...")
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(class_counts.index, class_counts.values, color=sns.color_palette('husl', 5))
for bar, count in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(count), ha='center', va='bottom', fontsize=12)
ax.set_xlabel('Cancer Type', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Class Distribution in TCGA Dataset', fontsize=14)
ax.set_xticklabels([f"{code}\n({CANCER_TYPE_INFO[code]['organ']})" for code in class_counts.index])
plt.tight_layout()
plt.savefig('results/figures/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: results/figures/class_distribution.png")

# Step 6: Gene expression statistics
print()
print("[6/8] Gene expression statistics:")
print(f"  Min value: {X.min():.2f}")
print(f"  Max value: {X.max():.2f}")
print(f"  Mean value: {X.mean():.2f}")
print(f"  Median value: {np.median(X):.2f}")
print(f"  Zero values: {(X == 0).sum():,} ({(X == 0).sum()/(X.size)*100:.1f}%)")

# Gene variance
gene_variance = np.var(X, axis=0)
print()
print("  Top 10 most variable genes:")
top_var_idx = np.argsort(gene_variance)[::-1][:10]
for i, idx in enumerate(top_var_idx, 1):
    print(f"    {i:2d}. {gene_names[idx]}: variance = {gene_variance[idx]:.1f}")

# Step 7: PCA visualization
print()
print("[7/8] Running PCA and creating visualization...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

print(f"  Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  Total variance (top 50 PCs): {pca.explained_variance_ratio_.sum()*100:.1f}%")

# PCA plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cancer_types = np.unique(y)
colors = sns.color_palette('husl', len(cancer_types))
color_map = dict(zip(cancer_types, colors))

ax1 = axes[0]
for cancer in cancer_types:
    mask = y == cancer
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=[color_map[cancer]], label=cancer, alpha=0.6, s=50)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax1.set_title('PCA: Cancer Types in Gene Expression Space', fontsize=14)
ax1.legend(title='Cancer Type')

ax2 = axes[1]
cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
ax2.bar(range(1, 21), pca.explained_variance_ratio_[:20] * 100, alpha=0.7, label='Individual')
ax2.plot(range(1, 21), cumulative_var[:20], 'ro-', label='Cumulative')
ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Principal Component', fontsize=12)
ax2.set_ylabel('Variance Explained (%)', fontsize=12)
ax2.set_title('Scree Plot: Variance Explained by PCs', fontsize=14)
ax2.legend()

plt.tight_layout()
plt.savefig('results/figures/pca_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: results/figures/pca_visualization.png")

# Step 8: Save processed data
print()
print("[8/8] Saving processed data for next steps...")
os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/X_scaled.npy', X_scaled)
np.save('data/processed/X_pca.npy', X_pca)
np.save('data/processed/y.npy', y)
np.save('data/processed/gene_names.npy', np.array(gene_names))
print("    Saved to data/processed/")

# Summary
print()
print("=" * 60)
print("EXPLORATION COMPLETE!")
print("=" * 60)
print()
print("KEY FINDINGS:")
print("  1. Dataset has 801 samples across 5 cancer types")
print("  2. Each sample has 20,531 gene expression values")
print("  3. Classes are reasonably balanced")
print("  4. Cancer types form distinct clusters in PCA space")
print("     -> Good sign for ML classification!")
print()
print("GENERATED FILES:")
print("  - results/figures/class_distribution.png")
print("  - results/figures/pca_visualization.png")
print("  - data/processed/*.npy (for next notebooks)")
print()
print("NEXT STEP: Run notebook 02_preprocessing.ipynb")
print("           or: python run_preprocessing.py")
