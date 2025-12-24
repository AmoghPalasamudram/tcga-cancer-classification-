# TCGA Cancer Classification: A Complete Learning Guide

## Author: Amogh Palasamudram
## Project: ML + Biology Cancer Classification

---

# TABLE OF CONTENTS

1. [Part 1: The Biology](#part-1-the-biology)
2. [Part 2: The Data](#part-2-the-data)
3. [Part 3: The Machine Learning Pipeline](#part-3-the-machine-learning-pipeline)
4. [Part 4: Walking Through the Code](#part-4-walking-through-the-code)
5. [Part 5: Deep Dive - The Mathematics](#part-5-deep-dive---the-mathematics)
6. [Part 6: Deep Dive - Cancer Biology](#part-6-deep-dive---cancer-biology)
7. [Part 7: Interpreting Results](#part-7-interpreting-results)
8. [Part 8: Key Concepts Summary](#part-8-key-concepts-summary)
9. [Part 9: Further Learning Resources](#part-9-further-learning-resources)

---

# PART 1: THE BIOLOGY

## 1.1 What is Cancer?

Cancer is a disease where cells in your body start dividing uncontrollably. Normally, your cells grow, divide, and die in an orderly way. Cancer cells ignore these rules.

**Normal Cell Cycle:**
```
Cell grows → Cell divides → Old cell dies → Balance maintained
```

**Cancer Cell Cycle:**
```
Cell grows → Cell divides → Cell DOESN'T die → Keeps dividing → Tumor forms
```

Different organs can develop cancer - breast, lung, kidney, colon, prostate, etc. Each cancer type:
- Behaves differently
- Has different genetic mutations
- Requires different treatments
- Has different survival rates

## 1.2 What is DNA and Genes?

**DNA (Deoxyribonucleic Acid):**
- The "instruction manual" for building and running your body
- Found in almost every cell
- Made of 4 chemical bases: A, T, G, C
- Human DNA has ~3 billion base pairs

**Genes:**
- Segments of DNA that code for specific proteins
- Humans have ~20,000 genes
- Each gene is like a recipe for making one protein
- Proteins do the actual work in your cells

```
DNA Structure (simplified):

    Gene 1        Gene 2        Gene 3
|------------|------------|------------|
ATCGATCGATCG GCTAGCTAGCTA TACGTACGTACG
|------------|------------|------------|
```

## 1.3 What is Gene Expression?

**The Central Dogma of Biology:**
```
DNA → RNA → Protein
     (transcription) (translation)
```

**Gene Expression** is the process of turning a gene "on" to make its protein.

- Not all 20,000 genes are active at once
- Different cells express different genes
- A liver cell expresses liver-specific genes
- A brain cell expresses brain-specific genes

**Think of it like a sound mixer:**
```
Gene A: ████████░░ (80% - highly expressed)
Gene B: ██░░░░░░░░ (20% - lowly expressed)
Gene C: █████░░░░░ (50% - moderately expressed)
Gene D: ░░░░░░░░░░ (0% - not expressed)
```

**Why it matters for cancer:**
- Cancer cells have ABNORMAL gene expression patterns
- Some genes that should be "off" are turned "on" (oncogenes)
- Some genes that should be "on" are turned "off" (tumor suppressors)
- These patterns are different for each cancer type = "molecular fingerprint"

## 1.4 How We Measure Gene Expression

**RNA Sequencing (RNA-seq):**

1. Extract RNA from tumor sample
2. Convert RNA to DNA (easier to work with)
3. Sequence it (read the letters)
4. Count how many times each gene appears
5. Higher count = higher expression

**The Output:**
```
Gene Name    | Read Count | Normalized Value
-------------|------------|------------------
BRCA1        | 15,234     | 8.2
TP53         | 8,456      | 6.1
MYC          | 52,123     | 12.4
...          | ...        | ...
```

The normalized values (usually log-transformed) are what we use for machine learning.

---

# PART 2: THE DATA

## 2.1 The TCGA Dataset

**TCGA = The Cancer Genome Atlas**
- Massive NIH-funded project
- Collected tumor samples from thousands of patients
- Sequenced their DNA, RNA, proteins
- Made it all publicly available for research

**Our specific dataset:**
- Source: UCI Machine Learning Repository (curated subset)
- 801 tumor samples
- 5 cancer types
- 20,531 genes measured per sample

## 2.2 Data Structure

```
Our data matrix (X):

              Gene_1  Gene_2  Gene_3  ...  Gene_20531
Sample_001     5.2     0.0     8.1   ...     2.1
Sample_002     2.1     6.3     1.2   ...     0.5
Sample_003     5.5     0.1     7.9   ...     2.3
...            ...     ...     ...   ...     ...
Sample_801     1.8     7.1     0.9   ...     0.8

Shape: 801 rows (samples) x 20,531 columns (genes)
```

**Labels (y):**
```
Sample_001 → BRCA (Breast Cancer)
Sample_002 → LUAD (Lung Cancer)
Sample_003 → BRCA (Breast Cancer)
...
Sample_801 → KIRC (Kidney Cancer)
```

## 2.3 The Five Cancer Types

| Code | Full Name | Organ | Samples | % of Data |
|------|-----------|-------|---------|-----------|
| BRCA | Breast Invasive Carcinoma | Breast | 300 | 37.5% |
| KIRC | Kidney Renal Clear Cell Carcinoma | Kidney | 146 | 18.2% |
| LUAD | Lung Adenocarcinoma | Lung | 141 | 17.6% |
| PRAD | Prostate Adenocarcinoma | Prostate | 136 | 17.0% |
| COAD | Colon Adenocarcinoma | Colon | 78 | 9.7% |

**Clinical Notes:**
- **BRCA**: Most common cancer in women; hormone-driven
- **KIRC**: Most common kidney cancer; often linked to VHL gene mutations
- **LUAD**: Lung cancer type common in non-smokers; EGFR mutations common
- **PRAD**: Most common cancer in men; hormone (androgen) driven
- **COAD**: Colorectal cancer; often linked to APC gene mutations

---

# PART 3: THE MACHINE LEARNING PIPELINE

## 3.1 What is Machine Learning?

Machine Learning (ML) is teaching computers to find patterns in data and make predictions without being explicitly programmed.

**Traditional Programming:**
```
Input + Rules → Output
(data)  (human-written)
```

**Machine Learning:**
```
Input + Output → Rules
(data)  (examples)  (learned by computer)
```

## 3.2 Types of Machine Learning

**Supervised Learning** (What we're doing):
- We have labeled examples
- Model learns mapping: Input → Output
- Examples: Classification, Regression

**Unsupervised Learning:**
- No labels
- Model finds patterns/structure
- Examples: Clustering, Dimensionality Reduction

**Our Task: Multi-class Classification**
```
Input: 20,531 gene expression values
Output: One of 5 cancer types
```

## 3.3 The ML Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: DATA COLLECTION                                            │
│  - Gather gene expression data from TCGA                            │
│  - 801 samples, 20,531 features                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: EXPLORATORY DATA ANALYSIS (EDA)                            │
│  - Understand distributions                                          │
│  - Check for missing values                                          │
│  - Visualize patterns (PCA)                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: PREPROCESSING                                               │
│  - Train/test split                                                  │
│  - Feature selection (20,531 → 1,000)                               │
│  - Feature scaling (standardization)                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: MODEL TRAINING                                              │
│  - Choose algorithms                                                 │
│  - Train on training data                                            │
│  - Tune hyperparameters                                              │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: EVALUATION                                                  │
│  - Test on held-out data                                             │
│  - Calculate metrics (accuracy, precision, recall)                   │
│  - Analyze errors                                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6: INTERPRETATION                                              │
│  - Feature importance                                                │
│  - Biological meaning                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

# PART 4: WALKING THROUGH THE CODE

## 4.1 Data Loading (data_loader.py)

```python
X, y, gene_names, sample_ids = load_tcga_data()
```

**What each variable contains:**
- `X`: NumPy array of shape (801, 20531) - the gene expression matrix
- `y`: NumPy array of shape (801,) - cancer type labels
- `gene_names`: List of 20,531 gene identifiers
- `sample_ids`: List of 801 sample identifiers

## 4.2 Exploration (run_exploration.py)

**Purpose:** Understand the data before modeling

**Key analyses:**

1. **Class Distribution**
   - Check if classes are balanced
   - Imbalanced data can bias models

2. **Expression Statistics**
   - Min, max, mean values
   - Percentage of zeros (sparse data)

3. **PCA Visualization**
   - Reduce 20,531 dimensions to 2
   - See if cancer types naturally cluster
   - Good separation = good sign for ML

## 4.3 Preprocessing (run_preprocessing.py)

**Step 1: Train/Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
- 80% for training (640 samples)
- 20% for testing (161 samples)
- `stratify=y`: Maintains class proportions in both sets
- `random_state=42`: Reproducibility

**Why split?**
- Training data: Model learns patterns
- Test data: Evaluate on "unseen" data
- Prevents overfitting (memorizing vs learning)

**Step 2: Feature Selection**

Problem: 20,531 features >> 801 samples (curse of dimensionality)

Solution 1 - Variance Filtering:
```python
# Remove genes with low variance
# If a gene has same value for all samples, it's useless
VarianceThreshold(threshold=median_variance)
# 20,531 → 10,265 genes
```

Solution 2 - SelectKBest with ANOVA F-test:
```python
# Keep genes most associated with cancer type
SelectKBest(score_func=f_classif, k=1000)
# 10,265 → 1,000 genes
```

**Step 3: Feature Scaling**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Standardization formula:
```
z = (x - μ) / σ

Where:
- x = original value
- μ = mean of the feature
- σ = standard deviation of the feature
- z = standardized value (mean=0, std=1)
```

**Why scale?**
- Different genes have different ranges
- Many algorithms assume similar scales
- Prevents features with large values from dominating

## 4.4 Model Training (run_training.py)

**Cross-Validation:**
```python
cv = StratifiedKFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, X_train, y_train, cv=cv)
```

5-Fold Cross-Validation:
```
Fold 1: [TEST][train][train][train][train] → Score 1
Fold 2: [train][TEST][train][train][train] → Score 2
Fold 3: [train][train][TEST][train][train] → Score 3
Fold 4: [train][train][train][TEST][train] → Score 4
Fold 5: [train][train][train][train][TEST] → Score 5

Final Score = Average of all 5 scores
```

**Why cross-validation?**
- More reliable than single train/test split
- Uses all data for both training and testing
- Reduces variance in performance estimate

---

# PART 5: DEEP DIVE - THE MATHEMATICS

## 5.1 Logistic Regression

Despite its name, Logistic Regression is a **classification** algorithm.

**Binary Logistic Regression (2 classes):**

The model learns a linear combination of features:
```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
z = w₀ + Σ(wᵢxᵢ)
```

Where:
- xᵢ = feature values (gene expression)
- wᵢ = learned weights
- w₀ = bias term

The sigmoid function converts z to probability:
```
σ(z) = 1 / (1 + e^(-z))

Output range: 0 to 1 (probability)
```

**Sigmoid Function Visualization:**
```
P(y=1)
  1.0 |                    ___________
      |                 __/
  0.5 |              __/
      |           __/
  0.0 |__________/
      +-----------------------------→ z
         -∞        0        +∞
```

**Decision Rule:**
```
If P(y=1) > 0.5 → Predict class 1
If P(y=1) ≤ 0.5 → Predict class 0
```

**Multinomial Logistic Regression (multiple classes):**

For K classes, we use the softmax function:
```
P(y=k|x) = e^(zₖ) / Σⱼe^(zⱼ)

Where:
- zₖ = w₀ₖ + Σ(wᵢₖxᵢ) for each class k
- The denominator sums over all K classes
- Output: K probabilities that sum to 1
```

**Example for our 5 cancer types:**
```
Input: Gene expression values
Model computes: z_BRCA, z_COAD, z_KIRC, z_LUAD, z_PRAD

Softmax converts to probabilities:
P(BRCA) = 0.85  ← Highest
P(COAD) = 0.02
P(KIRC) = 0.05
P(LUAD) = 0.06
P(PRAD) = 0.02

Prediction: BRCA (highest probability)
```

**Loss Function (Cross-Entropy):**
```
L = -Σᵢ Σₖ yᵢₖ log(pᵢₖ)

Where:
- yᵢₖ = 1 if sample i belongs to class k, else 0
- pᵢₖ = predicted probability for sample i, class k
```

**Training (Gradient Descent):**
```
1. Initialize weights randomly
2. For each iteration:
   a. Calculate predictions
   b. Calculate loss
   c. Calculate gradients: ∂L/∂w
   d. Update weights: w = w - α * ∂L/∂w
3. Repeat until convergence

α = learning rate (step size)
```

**Why Logistic Regression worked best for our data:**
- Gene expression patterns are roughly linearly separable
- High-dimensional data (1000 features) - linear models work well
- Less prone to overfitting than complex models
- Fast and interpretable

## 5.2 Random Forest

Random Forest is an **ensemble** method using multiple decision trees.

**Decision Tree Basics:**

A decision tree makes predictions through a series of if-then rules:
```
                    [Gene_A > 5.2?]
                    /            \
                 Yes              No
                 /                  \
        [Gene_B > 3.1?]        [Gene_C > 7.8?]
         /        \              /        \
       Yes        No           Yes        No
       /           \            /           \
    BRCA         LUAD        KIRC         PRAD
```

**How splits are chosen (for classification):**

Gini Impurity:
```
Gini(S) = 1 - Σₖ(pₖ)²

Where pₖ = proportion of class k in set S

Perfect purity (all one class): Gini = 0
Maximum impurity (equal mix): Gini = 0.8 (for 5 classes)
```

Information Gain (alternative):
```
IG(S, A) = Entropy(S) - Σᵥ (|Sᵥ|/|S|) * Entropy(Sᵥ)

Entropy(S) = -Σₖ pₖ log₂(pₖ)
```

The algorithm chooses the split that maximizes information gain (or minimizes Gini).

**Random Forest Algorithm:**

```
1. Create B bootstrap samples from training data
   (random sampling WITH replacement)

2. For each bootstrap sample b:
   a. Grow a decision tree Tᵦ
   b. At each node, randomly select m features (m << total features)
   c. Find best split among those m features
   d. Grow tree to maximum depth (no pruning)

3. Prediction by majority voting:
   ŷ = mode({T₁(x), T₂(x), ..., Tᵦ(x)})
```

**Key hyperparameters:**
- `n_estimators`: Number of trees (we used 200)
- `max_depth`: Maximum depth of each tree (we used 20)
- `max_features`: Features considered at each split (default: √n)
- `min_samples_split`: Minimum samples to split a node

**Why "Random"?**
1. Bootstrap sampling: Each tree sees different data
2. Feature randomization: Each split considers different features
3. This diversity reduces overfitting!

**Feature Importance in Random Forest:**

Calculated by measuring how much each feature decreases impurity:
```
Importance(feature j) = Σ (weighted impurity decrease at all nodes using feature j)

Normalized so all importances sum to 1
```

## 5.3 Support Vector Machine (SVM)

SVM finds the optimal hyperplane that separates classes.

**Linear SVM (2 classes):**

Goal: Find hyperplane w·x + b = 0 that maximizes the margin.

```
                    w·x + b = +1  (positive class boundary)
    +    +         ---------------
         +        /
    +            /  ← Margin (maximize this!)
                /
    -    -     /
         -    ---------------
    -          w·x + b = -1  (negative class boundary)
```

**Optimization Problem:**
```
Minimize:    (1/2)||w||²
Subject to:  yᵢ(w·xᵢ + b) ≥ 1  for all i

Where:
- ||w|| = magnitude of weight vector
- yᵢ = class label (+1 or -1)
- Margin = 2/||w||
```

**Support Vectors:**
- Points exactly on the margin boundaries
- These "support" the hyperplane
- Only these points matter for the decision boundary!

**Soft Margin SVM (allowing some errors):**
```
Minimize:    (1/2)||w||² + C Σᵢ ξᵢ
Subject to:  yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
             ξᵢ ≥ 0

Where:
- ξᵢ = slack variables (allow misclassification)
- C = regularization parameter (tradeoff between margin and errors)
  - Large C: Less tolerant of errors (might overfit)
  - Small C: More tolerant of errors (might underfit)
```

**Kernel Trick (for non-linear boundaries):**

When data isn't linearly separable, we map it to higher dimensions:
```
Original 2D:          After kernel mapping:
    - - -                    + + +
  -   +   -              +         +
    + + +           -       - - -       -
  -   +   -              +         +
    - - -                    + + +

Not separable!           Now separable!
```

**RBF (Radial Basis Function) Kernel:**
```
K(xᵢ, xⱼ) = exp(-γ ||xᵢ - xⱼ||²)

Where:
- γ (gamma) controls the "reach" of each training example
- Large γ: Each point has small influence (complex boundary)
- Small γ: Each point has large influence (smooth boundary)
```

**Multi-class SVM:**
- SVM is inherently binary
- For K classes, use "One-vs-Rest" or "One-vs-One"
- One-vs-Rest: Train K classifiers, each separating one class from all others
- Prediction: Class with highest confidence score

## 5.4 Evaluation Metrics Mathematics

**Confusion Matrix:**
```
                    Predicted
                 Pos    Neg
Actual  Pos  [   TP     FN   ]
        Neg  [   FP     TN   ]

TP = True Positive (correctly predicted positive)
TN = True Negative (correctly predicted negative)
FP = False Positive (incorrectly predicted positive) - Type I Error
FN = False Negative (incorrectly predicted negative) - Type II Error
```

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

"What fraction of all predictions were correct?"
```

**Precision:**
```
Precision = TP / (TP + FP)

"Of all positive predictions, how many were actually positive?"
"How trustworthy are positive predictions?"
```

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)

"Of all actual positives, how many did we find?"
"How completely did we find all positives?"
```

**F1-Score:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Harmonic mean of precision and recall
Balances both metrics
```

**ROC Curve and AUC:**

ROC = Receiver Operating Characteristic

```
True Positive Rate (TPR) = TP / (TP + FN) = Recall
False Positive Rate (FPR) = FP / (FP + TN)
```

ROC curve plots TPR vs FPR at different classification thresholds:
```
TPR
  1 |    ___________
    |   /
    |  /
    | /
  0 |/_______________
    0              1  FPR
```

AUC = Area Under ROC Curve
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random (something's wrong!)

---

# PART 6: DEEP DIVE - CANCER BIOLOGY

## 6.1 Molecular Basis of Cancer

**The Hallmarks of Cancer:**

Cancer cells acquire specific capabilities:

1. **Sustaining Proliferative Signaling**
   - Normal cells need external signals to divide
   - Cancer cells produce their own signals or become hypersensitive

2. **Evading Growth Suppressors**
   - Normal cells have "brakes" (tumor suppressors like p53, Rb)
   - Cancer cells disable these brakes

3. **Resisting Cell Death**
   - Normal cells undergo apoptosis (programmed death) when damaged
   - Cancer cells ignore death signals

4. **Enabling Replicative Immortality**
   - Normal cells can only divide ~50 times (Hayflick limit)
   - Cancer cells express telomerase, becoming immortal

5. **Inducing Angiogenesis**
   - Tumors need blood supply
   - Cancer cells signal for new blood vessel growth

6. **Activating Invasion & Metastasis**
   - Cancer cells can detach, travel, and colonize distant sites

## 6.2 Cancer-Specific Gene Expression

**Why different cancers have different expression patterns:**

1. **Cell of Origin**
   - Breast cells normally express breast-specific genes
   - These patterns persist in breast cancer
   - A breast tumor still "looks like" breast tissue at the molecular level

2. **Driver Mutations**
   - Each cancer type has characteristic mutations
   - BRCA1/2 mutations in breast cancer
   - EGFR mutations in lung adenocarcinoma
   - VHL mutations in kidney cancer

3. **Tissue-Specific Pathways**
   - Hormone receptor pathways in breast/prostate cancer
   - Surfactant genes in lung cancer
   - Digestive enzyme genes in colon cancer

## 6.3 Understanding Each Cancer Type in Our Dataset

**BRCA (Breast Invasive Carcinoma):**
```
Key molecular features:
- Estrogen/Progesterone receptor expression (ER/PR)
- HER2 amplification in some subtypes
- BRCA1/2 mutations in hereditary cases

Characteristic genes:
- ESR1 (estrogen receptor)
- PGR (progesterone receptor)
- ERBB2 (HER2)
- Mammary-specific keratins (KRT8, KRT18)
```

**KIRC (Kidney Renal Clear Cell Carcinoma):**
```
Key molecular features:
- VHL tumor suppressor loss (>90% of cases)
- HIF pathway activation (hypoxia response)
- Clear cytoplasm (lipid/glycogen accumulation)

Characteristic genes:
- CA9 (carbonic anhydrase 9) - marker for VHL loss
- VEGF (vascular endothelial growth factor)
- Kidney-specific transporters
```

**LUAD (Lung Adenocarcinoma):**
```
Key molecular features:
- EGFR mutations
- KRAS mutations
- ALK fusions

Characteristic genes:
- Surfactant proteins (SFTPA1, SFTPB)
- TTF-1 (NKX2-1) transcription factor
- Mucin genes
```

**PRAD (Prostate Adenocarcinoma):**
```
Key molecular features:
- Androgen receptor (AR) signaling
- TMPRSS2-ERG fusion (50% of cases)
- PTEN loss

Characteristic genes:
- AR (androgen receptor)
- KLK3 (PSA - prostate specific antigen)
- Prostate-specific membrane antigen (FOLH1)
```

**COAD (Colon Adenocarcinoma):**
```
Key molecular features:
- APC tumor suppressor loss
- WNT pathway activation
- Microsatellite instability in some cases

Characteristic genes:
- CDX2 (intestinal transcription factor)
- Mucins (MUC2)
- Digestive enzymes
- Intestinal keratins (KRT20)
```

## 6.4 Why Machine Learning Works for Cancer Classification

**The Biological Foundation:**

1. **Distinct Molecular Fingerprints**
   - Each cancer maintains cell-of-origin expression patterns
   - Creates natural separation in gene expression space

2. **Consistent Patterns**
   - Despite patient-to-patient variation
   - Core cancer-specific patterns are preserved

3. **High-Dimensional Signatures**
   - Not just one gene, but thousands
   - Pattern of many genes together is informative

**What the PCA Plot Showed:**
```
- KIRC (kidney) forms distinct cluster on the right
  → Kidney cells have very different expression profile

- BRCA (breast) and PRAD (prostate) cluster together
  → Both are hormone-driven epithelial cancers

- Each cancer occupies its own region in "gene space"
```

---

# PART 7: INTERPRETING RESULTS

## 7.1 Understanding Model Performance

**Our Results:**
```
Model                 CV Accuracy    Test Accuracy    ROC-AUC
Logistic Regression   99.84%         99.38%          1.000
Random Forest         99.69%         98.76%          1.000
SVM (RBF)            99.69%         99.38%          1.000
```

**What this means:**
- All models performed excellently (>98% accuracy)
- The problem is "easy" for ML (clear biological signal)
- Logistic Regression won - simpler model, less overfitting

**The Single Misclassification:**
```
1 Lung cancer sample was predicted as Breast cancer

Possible explanations:
- Sample quality issues
- Unusual molecular subtype
- Biological similarity between these samples
- Inherent overlap at molecular level
```

## 7.2 Interpreting Feature Importance

**What Feature Importance Tells Us:**

The most important genes are those that best distinguish cancer types.

```
Top genes from our analysis:
1. gene_7964
2. gene_17801
3. gene_15898
...
```

**Biological Interpretation Steps:**

1. **Map gene IDs to gene symbols**
   - Our data uses numeric IDs
   - Need to convert to standard gene names (e.g., BRCA1, TP53)

2. **Literature search**
   - Are these known cancer markers?
   - Tissue-specific genes?
   - Oncogenes or tumor suppressors?

3. **Pathway analysis**
   - What biological processes are these genes involved in?
   - Are they part of known cancer pathways?

4. **Validation**
   - Do other studies find similar important genes?
   - Can these be validated experimentally?

## 7.3 How to Look Up Gene Functions

**Online Resources:**

1. **GeneCards** (www.genecards.org)
   - Comprehensive gene information
   - Expression patterns, pathways, diseases

2. **NCBI Gene** (www.ncbi.nlm.nih.gov/gene)
   - Official gene database
   - Literature references

3. **UniProt** (www.uniprot.org)
   - Protein function information
   - Structure and domains

4. **KEGG Pathways** (www.kegg.jp)
   - Metabolic and signaling pathways
   - Disease associations

**Example Analysis:**
```
If top gene is ESR1 (Estrogen Receptor 1):
- Known: Major driver in breast cancer
- Makes sense: Breast cancer (BRCA) is in our dataset
- Interpretation: Model correctly identified biologically relevant marker
```

## 7.4 Confusion Matrix Analysis

**Our Confusion Matrix:**
```
              Predicted:
              BRCA  COAD  KIRC  LUAD  PRAD
Actual BRCA [  60     0     0     0     0  ]
       COAD [   0    16     0     0     0  ]
       KIRC [   0     0    30     0     0  ]
       LUAD [   1     0     0    27     0  ]
       PRAD [   0     0     0     0    27  ]
```

**Interpretation:**
- Diagonal = correct predictions
- Off-diagonal = errors
- LUAD→BRCA: 1 lung cancer misclassified as breast cancer
- All other predictions perfect

**Per-Class Analysis:**
```
BRCA: 60/60 correct (100%) - Perfect
COAD: 16/16 correct (100%) - Perfect
KIRC: 30/30 correct (100%) - Perfect
LUAD: 27/28 correct (96.4%) - 1 error
PRAD: 27/27 correct (100%) - Perfect
```

## 7.5 ROC Curves Analysis

**What Our ROC Curves Show:**
```
All curves hug the top-left corner
AUC = 1.0 for all classes
```

**Interpretation:**
- Model has perfect discrimination ability
- At any threshold, can perfectly separate classes
- The gene expression signal is very strong

**If AUC were lower (hypothetically):**
```
AUC = 0.9: Very good
AUC = 0.8: Good
AUC = 0.7: Fair
AUC = 0.6: Poor
AUC = 0.5: No discrimination (random)
```

---

# PART 8: KEY CONCEPTS SUMMARY

## 8.1 Biology Terms

| Term | Definition |
|------|------------|
| **Gene** | Segment of DNA that codes for a protein |
| **Gene Expression** | The process of turning a gene "on" to make protein |
| **RNA-seq** | Technology to measure gene expression |
| **Oncogene** | Gene that promotes cancer when activated |
| **Tumor Suppressor** | Gene that prevents cancer; must be inactivated |
| **Biomarker** | Measurable indicator of biological state |

## 8.2 Machine Learning Terms

| Term | Definition |
|------|------------|
| **Feature** | Input variable (gene expression value) |
| **Label** | Output variable (cancer type) |
| **Training** | Teaching model with labeled examples |
| **Testing** | Evaluating model on unseen data |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Cross-validation** | Repeated train/test for reliable evaluation |
| **Hyperparameter** | Model setting chosen before training |

## 8.3 Evaluation Terms

| Term | Definition |
|------|------------|
| **Accuracy** | Fraction of correct predictions |
| **Precision** | Of positive predictions, how many correct |
| **Recall** | Of actual positives, how many found |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC Curve** | Plot of TPR vs FPR at different thresholds |
| **AUC** | Area under ROC curve (higher = better) |

## 8.4 Common Pitfalls

1. **Data Leakage**
   - Preprocessing before train/test split
   - Test data information leaking into training

2. **Overfitting**
   - Too many features, too few samples
   - Model too complex

3. **Class Imbalance**
   - Accuracy misleading with imbalanced classes
   - Need stratified sampling

4. **Not Understanding Biology**
   - ML can find patterns that aren't meaningful
   - Always validate with biological knowledge

---

# PART 9: FURTHER LEARNING RESOURCES

## 9.1 Machine Learning

**Books:**
- "An Introduction to Statistical Learning" - James et al. (Free PDF)
- "Pattern Recognition and Machine Learning" - Bishop
- "Hands-On Machine Learning" - Geron

**Online Courses:**
- Andrew Ng's Machine Learning (Coursera)
- Fast.ai Practical Deep Learning
- StatQuest YouTube Channel (excellent visual explanations)

## 9.2 Bioinformatics

**Books:**
- "Bioinformatics and Functional Genomics" - Pevsner
- "Statistical Methods in Bioinformatics" - Ewens & Grant

**Online Resources:**
- Coursera: Genomic Data Science Specialization
- edX: Data Analysis for Life Sciences
- Rosalind (rosalind.info) - Bioinformatics practice problems

## 9.3 Cancer Genomics

**Databases:**
- TCGA Portal (portal.gdc.cancer.gov)
- cBioPortal (cbioportal.org)
- COSMIC (cancer.sanger.ac.uk/cosmic)

**Literature:**
- "The Hallmarks of Cancer" - Hanahan & Weinberg (seminal paper)
- Nature Reviews Cancer journal
- Cancer Discovery journal

## 9.4 Python for Data Science

**Libraries to Learn:**
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - Machine learning
- matplotlib/seaborn - Visualization
- Biopython - Bioinformatics tools

**Practice:**
- Kaggle competitions
- UCI ML Repository datasets
- GitHub bioinformatics projects

---

# CONCLUSION

You've built a complete cancer classification pipeline that:

1. **Downloads** real TCGA gene expression data
2. **Explores** the data to understand patterns
3. **Preprocesses** with proper feature selection
4. **Trains** multiple ML models
5. **Evaluates** with rigorous metrics
6. **Interprets** results biologically

**Key Achievement:** 99.4% accuracy distinguishing 5 cancer types from gene expression

**What You Learned:**
- How gene expression differs between cancer types
- How to handle high-dimensional biological data
- How to apply and compare ML classification algorithms
- How to evaluate and interpret model results

**Next Steps:**
- Look up what the top genes actually are
- Try different feature selection methods
- Explore deep learning approaches
- Apply similar methods to other biological problems

---

*This guide was created as part of the TCGA Cancer Classification project.*
*Author: Amogh Palasamudram*
