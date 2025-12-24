"""
TCGA Pan-Cancer Model Training Script
Run after run_preprocessing.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("TCGA PAN-CANCER MODEL TRAINING")
print("=" * 60)
print()

# Step 1: Import libraries
print("[1/7] Importing libraries...")
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("    Note: XGBoost not available, skipping")

plt.style.use('seaborn-v0_8-whitegrid')
RANDOM_STATE = 42
print("    Done!")

# Step 2: Load data
print()
print("[2/7] Loading preprocessed data...")
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

with open('data/processed/selected_genes.pkl', 'rb') as f:
    selected_genes = pickle.load(f)

with open('data/processed/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

class_names = label_encoder.classes_
n_classes = len(class_names)

print(f"    Training: {X_train.shape}")
print(f"    Test: {X_test.shape}")
print(f"    Classes: {list(class_names)}")

# Step 3: Define models
print()
print("[3/7] Defining models...")
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        multi_class='multinomial',
        solver='lbfgs'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=RANDOM_STATE
    )
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0
    )

for name in models:
    print(f"    - {name}")

# Step 4: Cross-validation
print()
print("[4/7] Running 5-fold cross-validation...")
print("-" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, model in models.items():
    print(f"    {name}...", end=" ", flush=True)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

print("-" * 50)

# Step 5: Train final models and evaluate on test set
print()
print("[5/7] Training on full training set & evaluating on test set...")
print("-" * 60)

trained_models = {}
test_results = {}

for name, model in models.items():
    print(f"    {name}...", end=" ", flush=True)

    model.fit(X_train, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    roc_auc = roc_auc_score(y_test_bin, y_prob, average='weighted', multi_class='ovr')

    test_results[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print(f"Test Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

print("-" * 60)

# Find best model
best_model_name = max(test_results, key=lambda x: test_results[x]['accuracy'])
best_results = test_results[best_model_name]

# Step 6: Generate visualizations
print()
print("[6/7] Generating visualizations...")
os.makedirs('results/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 6a. CV Comparison boxplot
print("    - Cross-validation comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))
cv_df = pd.DataFrame(cv_results)
cv_df_melted = cv_df.melt(var_name='Model', value_name='Accuracy')
sns.boxplot(data=cv_df_melted, x='Model', y='Accuracy', palette='husl', ax=ax)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('')
ax.set_title('5-Fold Cross-Validation Results', fontsize=14)
ax.set_ylim([0.9, 1.01])
for i, name in enumerate(models.keys()):
    mean_acc = cv_results[name].mean()
    ax.text(i, mean_acc + 0.008, f'{mean_acc:.3f}', ha='center', fontsize=10)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('results/figures/cv_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 6b. Confusion Matrix
print("    - Confusion matrix...")
cm = confusion_matrix(y_test, best_results['y_pred'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax1)
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_title(f'Confusion Matrix ({best_model_name})', fontsize=14)

ax2 = axes[1]
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax2)
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xlabel('Predicted Label', fontsize=12)
ax2.set_title('Confusion Matrix (Normalized %)', fontsize=14)

plt.tight_layout()
plt.savefig('results/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 6c. ROC Curves
print("    - ROC curves...")
y_test_bin = label_binarize(y_test, classes=range(n_classes))
y_prob = best_results['y_prob']

fig, ax = plt.subplots(figsize=(10, 8))
colors = sns.color_palette('husl', n_classes)

for i, (color, class_name) in enumerate(zip(colors, class_names)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc_i = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{class_name} (AUC = {roc_auc_i:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curves - {best_model_name}', fontsize=14)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('results/figures/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# 6d. Feature Importance (Random Forest)
print("    - Feature importance plot...")
rf_model = trained_models['Random Forest']
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'gene': selected_genes,
    'importance': importances
}).sort_values('importance', ascending=False)

top_n = 25
top_features = importance_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(range(top_n), top_features['importance'].values[::-1], color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['gene'].values[::-1])
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title(f'Top {top_n} Most Important Genes for Cancer Classification', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Step 7: Save models and results
print()
print("[7/7] Saving models and results...")

# Save best model
model_path = f'models/{best_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_best.joblib'
joblib.dump(trained_models[best_model_name], model_path)
print(f"    Best model saved: {model_path}")

# Save results CSV
results_df = pd.DataFrame({
    'Model': list(test_results.keys()),
    'CV_Accuracy_Mean': [cv_results[m].mean() for m in test_results.keys()],
    'CV_Accuracy_Std': [cv_results[m].std() for m in test_results.keys()],
    'Test_Accuracy': [test_results[m]['accuracy'] for m in test_results.keys()],
    'Test_ROC_AUC': [test_results[m]['roc_auc'] for m in test_results.keys()]
})
results_df.to_csv('results/model_comparison.csv', index=False)
importance_df.to_csv('results/feature_importance.csv', index=False)
print("    Results saved to results/")

# Final Summary
print()
print("=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print()
print("MODEL COMPARISON:")
print("-" * 70)
print(f"{'Model':<25} {'CV Accuracy':>15} {'Test Accuracy':>15} {'ROC-AUC':>12}")
print("-" * 70)
for name in models.keys():
    cv_acc = cv_results[name].mean()
    test_acc = test_results[name]['accuracy']
    roc = test_results[name]['roc_auc']
    marker = " <-- BEST" if name == best_model_name else ""
    print(f"{name:<25} {cv_acc:>15.4f} {test_acc:>15.4f} {roc:>12.4f}{marker}")
print("-" * 70)

print()
print(f"BEST MODEL: {best_model_name}")
print(f"  Test Accuracy: {best_results['accuracy']:.2%}")
print(f"  Test ROC-AUC: {best_results['roc_auc']:.4f}")

print()
print("PER-CLASS PERFORMANCE:")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, best_results['y_pred'])
for i, class_name in enumerate(class_names):
    print(f"  {class_name}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1={f1[i]:.2f}")

print()
print("TOP 10 PREDICTIVE GENES:")
for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
    print(f"  {i:2d}. {row['gene']} (importance: {row['importance']:.4f})")

print()
print("GENERATED FILES:")
print("  - results/figures/cv_comparison.png")
print("  - results/figures/confusion_matrix.png")
print("  - results/figures/roc_curves.png")
print("  - results/figures/feature_importance.png")
print("  - results/model_comparison.csv")
print("  - results/feature_importance.csv")
print("  - models/*_best.joblib")
print()
print("=" * 70)
