"""
TCGA Pan-Cancer Preprocessing Script
Run after run_exploration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("TCGA DATA PREPROCESSING")
print("=" * 60)
print()

# Step 1: Import libraries
print("[1/6] Importing libraries...")
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from data_loader import load_tcga_data
print("    Done!")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Step 2: Load data
print()
print("[2/6] Loading data...")
X, y, gene_names, sample_ids = load_tcga_data(verbose=False)
print(f"    Loaded: {X.shape[0]} samples x {X.shape[1]:,} genes")

# Step 3: Encode labels and split
print()
print("[3/6] Splitting into train/test sets (80/20)...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("    Label encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"      {label} -> {i}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)
print(f"    Training set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")

# Step 4: Feature selection
print()
print("[4/6] Feature selection...")

# Variance filter
gene_variance = np.var(X_train, axis=0)
variance_threshold = np.percentile(gene_variance, 50)
var_selector = VarianceThreshold(threshold=variance_threshold)
X_train_var = var_selector.fit_transform(X_train)
X_test_var = var_selector.transform(X_test)

var_mask = var_selector.get_support()
genes_after_var = [g for g, m in zip(gene_names, var_mask) if m]
print(f"    After variance filter: {X_train_var.shape[1]:,} genes")

# SelectKBest with F-test
K_FEATURES = 1000
kbest_selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
X_train_selected = kbest_selector.fit_transform(X_train_var, y_train)
X_test_selected = kbest_selector.transform(X_test_var)

kbest_mask = kbest_selector.get_support()
selected_genes = [g for g, m in zip(genes_after_var, kbest_mask) if m]
print(f"    After SelectKBest: {X_train_selected.shape[1]} genes")

# Show top genes
f_scores = kbest_selector.scores_[kbest_mask]
gene_scores = sorted(zip(selected_genes, f_scores), key=lambda x: x[1], reverse=True)
print("    Top 10 discriminative genes:")
for gene, score in gene_scores[:10]:
    print(f"      {gene}: F-score = {score:.1f}")

# Step 5: Scaling
print()
print("[5/6] Standardizing features...")
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_selected)
X_test_final = scaler.transform(X_test_selected)
print(f"    Training mean: {X_train_final.mean():.6f} (should be ~0)")
print(f"    Training std: {X_train_final.std():.6f} (should be ~1)")

# Step 6: Save
print()
print("[6/6] Saving preprocessed data...")
os.makedirs('data/processed', exist_ok=True)

np.save('data/processed/X_train.npy', X_train_final)
np.save('data/processed/X_test.npy', X_test_final)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)

with open('data/processed/selected_genes.pkl', 'wb') as f:
    pickle.dump(selected_genes, f)

with open('data/processed/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('data/processed/preprocessors.pkl', 'wb') as f:
    pickle.dump({
        'var_selector': var_selector,
        'kbest_selector': kbest_selector,
        'scaler': scaler
    }, f)

print("    Saved all files to data/processed/")

# Summary
print()
print("=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)
print()
print("SUMMARY:")
print(f"  Original: {X.shape[1]:,} genes")
print(f"  After variance filter: {X_train_var.shape[1]:,} genes")
print(f"  After SelectKBest: {K_FEATURES} genes")
print()
print(f"  Training samples: {X_train_final.shape[0]}")
print(f"  Test samples: {X_test_final.shape[0]}")
print()
print("NEXT STEP: python run_training.py")
