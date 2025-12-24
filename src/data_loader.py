"""
Data loader for TCGA Pan-Cancer gene expression dataset.

This module provides functions to download and load the TCGA dataset
from the UCI Machine Learning Repository.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile


# TCGA Pan-Cancer dataset from UCI ML Repository
# This is a curated version with ~800 samples across 5 cancer types
# Good for learning - not too large, but real data
UCI_TCGA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"

# Alternative: Kaggle has larger datasets
# For a more comprehensive dataset, we'll use a subset of TCGA from a reliable source


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def download_tcga_data(data_dir: str = None, verbose: bool = True) -> str:
    """
    Download TCGA Pan-Cancer dataset.

    This downloads a curated subset from UCI ML Repository containing
    gene expression data for 5 cancer types:
    - BRCA (Breast Cancer)
    - KIRC (Kidney Renal Clear Cell Carcinoma)
    - LUAD (Lung Adenocarcinoma)
    - PRAD (Prostate Adenocarcinoma)
    - COAD (Colon Adenocarcinoma)

    Parameters
    ----------
    data_dir : str, optional
        Directory to save data. Defaults to project's data/raw folder.
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Path to the downloaded data directory.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / "TCGA-PANCAN-HiSeq-801x20531.tar.gz"

    if not tar_path.exists():
        if verbose:
            print(f"Downloading TCGA dataset from UCI ML Repository...")
            print(f"URL: {UCI_TCGA_URL}")

        urllib.request.urlretrieve(UCI_TCGA_URL, tar_path)

        if verbose:
            print(f"Download complete: {tar_path}")
    else:
        if verbose:
            print(f"Dataset already exists: {tar_path}")

    # Extract the tar.gz file
    import tarfile

    extract_dir = data_dir / "tcga_pancan"
    if not extract_dir.exists():
        if verbose:
            print("Extracting files...")

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)

        if verbose:
            print("Extraction complete!")

    return str(data_dir)


def load_tcga_data(data_dir: str = None, verbose: bool = True) -> tuple:
    """
    Load TCGA Pan-Cancer gene expression data.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing raw data.
    verbose : bool
        Print progress messages.

    Returns
    -------
    tuple
        (X, y, gene_names, sample_ids) where:
        - X: numpy array of gene expression values (samples x genes)
        - y: numpy array of cancer type labels
        - gene_names: list of gene names
        - sample_ids: list of sample identifiers
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"

    data_dir = Path(data_dir)

    # Check if data exists, if not download it
    data_file = data_dir / "TCGA-PANCAN-HiSeq-801x20531" / "data.csv"
    labels_file = data_dir / "TCGA-PANCAN-HiSeq-801x20531" / "labels.csv"

    if not data_file.exists():
        if verbose:
            print("Data not found. Downloading...")
        download_tcga_data(data_dir, verbose)

    if verbose:
        print("Loading gene expression data...")

    # Load expression data
    df_data = pd.read_csv(data_file)

    # Load labels
    df_labels = pd.read_csv(labels_file)

    if verbose:
        print(f"Loaded {df_data.shape[0]} samples with {df_data.shape[1]-1} genes")
        print(f"Cancer types: {df_labels['Class'].unique()}")

    # Extract features and labels
    sample_ids = df_data.iloc[:, 0].values
    X = df_data.iloc[:, 1:].values
    gene_names = df_data.columns[1:].tolist()
    y = df_labels['Class'].values

    return X, y, gene_names, sample_ids


def load_as_dataframe(data_dir: str = None) -> tuple:
    """
    Load TCGA data as pandas DataFrames.

    Returns
    -------
    tuple
        (df_expression, df_labels) DataFrames
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"

    data_dir = Path(data_dir)

    data_file = data_dir / "TCGA-PANCAN-HiSeq-801x20531" / "data.csv"
    labels_file = data_dir / "TCGA-PANCAN-HiSeq-801x20531" / "labels.csv"

    if not data_file.exists():
        download_tcga_data(data_dir)

    df_expression = pd.read_csv(data_file, index_col=0)
    df_labels = pd.read_csv(labels_file, index_col=0)

    return df_expression, df_labels


# Cancer type descriptions for interpretation
CANCER_TYPE_INFO = {
    "BRCA": {
        "full_name": "Breast Invasive Carcinoma",
        "organ": "Breast",
        "description": "Most common cancer in women worldwide"
    },
    "KIRC": {
        "full_name": "Kidney Renal Clear Cell Carcinoma",
        "organ": "Kidney",
        "description": "Most common type of kidney cancer in adults"
    },
    "LUAD": {
        "full_name": "Lung Adenocarcinoma",
        "organ": "Lung",
        "description": "Most common type of lung cancer, often in non-smokers"
    },
    "PRAD": {
        "full_name": "Prostate Adenocarcinoma",
        "organ": "Prostate",
        "description": "Most common cancer in men"
    },
    "COAD": {
        "full_name": "Colon Adenocarcinoma",
        "organ": "Colon",
        "description": "Third most common cancer worldwide"
    }
}


if __name__ == "__main__":
    # Test the data loader
    X, y, genes, samples = load_tcga_data(verbose=True)
    print(f"\nData shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of genes: {len(genes)}")
    print(f"Sample gene names: {genes[:5]}")
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cancer, count in zip(unique, counts):
        print(f"  {cancer}: {count} samples")
