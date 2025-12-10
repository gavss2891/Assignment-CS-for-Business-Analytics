# TV Product Duplicate Detection

Duplicate detection system for TV products using Locality Sensitive Hashing (LSH) with MinHash signatures and Multi-component Similarity Method MSM.

## Overview

This project implements an efficient duplicate detection pipeline that:
- Uses LSH to reduce pairwise comparisons
- Applies MSM computation on candidate pairs
- Performs agglomerative clustering for final grouping
- Compares performance with and without data correction (CleanMSMP+ vs MSMP+)

## Code Structure

- **`main.py`**: Main execution script with bootstrap evaluation and plotting
- **`lsh.py`**: LSH implementation (MinHash signatures, candidate pair generation)
- **`msm.py`**: Modified Similarity Measure for computing product similarity
- **`data_cleaning.py`**: Data preprocessing and normalization 
- **`TMWMSim.py`**: Title Model Word Similarity computation
- **`analyze_data.py`**: Data analysis utilities (Not a part of duplication detection workflow)

## Usage

### Basic Run

```python
python main.py
```

### Custom Parameters

Modify parameters in `main.py`:

```python
main_func(
    path="TVs-all-merged.json",
    bootstraps=8,
    gammas=[0.5, 0.6, 0.7, 0.75],
    epsilons=[0.4, 0.5, 0.6],
    mus=[0.6, 0.65, 0.7],
    fraction=0.5,
    compare=True  # Compare with/without data correction
)
```

### Output

- Performance metrics (F1, F1*, PC, PQ, Precision, Recall)
- Plots comparing CleanMSMP+ vs MSMP+ approaches
- Checkpoint files for resuming interrupted runs

