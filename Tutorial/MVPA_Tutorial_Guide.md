# MVPA and Representational Geometry Tutorial

## Welcome to Your First MVPA Analysis!

This tutorial will walk you through Multi-Voxel Pattern Analysis (MVPA) and representational geometry concepts. We'll start with the basics and build up to advanced analyses.

## Table of Contents
1. [What is MVPA and Why Use It?](#what-is-mvpa)
2. [Understanding Representational Geometry](#representational-geometry)
3. [Dimensionality Reduction Explained](#dimensionality-reduction)
4. [Pattern Distinctness Analysis](#pattern-distinctness)
5. [Representational Similarity Analysis (RSA)](#rsa)
6. [Cross-Validation and Statistical Controls](#cross-validation)
7. [Putting It All Together](#complete-analysis)

---

## What is MVPA and Why Use It? {#what-is-mvpa}

### Traditional fMRI Analysis (Univariate)
- Looks at each voxel independently
- Asks: "Is this voxel more active in condition A vs B?"
- Uses t-tests, ANOVAs on individual voxels
- **Problem**: Misses information encoded in patterns across voxels

### MVPA (Multivariate)
- Looks at patterns across multiple voxels simultaneously
- Asks: "What information is encoded in the pattern of activity?"
- Uses machine learning, pattern analysis
- **Advantage**: Can detect information even when individual voxels don't differ

### Simple Example
Imagine you have 5 voxels and two conditions:

```python
# Condition 1: Pattern [1, -1, 1, -1, 1]
# Condition 2: Pattern [-1, 1, -1, 1, -1]
```

**Univariate analysis**: Each voxel individually might not show significant differences
**MVPA analysis**: The overall patterns are clearly different!

This is why MVPA is powerful - it can detect information that traditional analysis misses.

---

## Understanding Representational Geometry {#representational-geometry}

### What is a Representational Space?

Think of the brain as creating a "map" where:
- Each stimulus/condition is a point in high-dimensional space
- Each dimension is a neuron/voxel
- Similar stimuli are close together
- Different stimuli are far apart

### Key Concepts

1. **Neural Population Vector**: The pattern of activity across all neurons for one stimulus
2. **Distance**: How different two neural patterns are
3. **Geometry**: The overall structure of how stimuli are arranged in neural space

### Why Does This Matter?

The geometry tells us about **neural coding**:
- Are similar choices represented similarly?
- How does value information map onto neural patterns?
- Do different brain regions use different coding schemes?

### Example: Delay Discounting
- **Sooner-smaller choices**: Might cluster together in neural space
- **Larger-later choices**: Might form a separate cluster
- **Value differences**: Might create gradients within clusters

---

## Dimensionality Reduction Explained {#dimensionality-reduction}

### The Problem
- fMRI data is **high-dimensional** (1000+ voxels per ROI)
- Hard to visualize and analyze
- Many dimensions might contain mostly noise

### The Solution: Dimensionality Reduction
Find a smaller number of dimensions that capture the important information.

### Principal Component Analysis (PCA)

**What it does**: Finds directions of maximum variance in your data

**How it works**:
1. Find the direction where data varies the most (PC1)
2. Find the next direction, perpendicular to PC1, with most remaining variance (PC2)
3. Continue until you have enough components

**Why it's useful**:
- Reduces noise
- Makes visualization possible
- Focuses on the most informative dimensions

### Code Example
```python
from sklearn.decomposition import PCA

# Original data: 100 trials × 1000 voxels
pca = PCA(n_components=10)  # Reduce to 10 dimensions
X_reduced = pca.fit_transform(X)  # Now: 100 trials × 10 components

# How much variance does each component explain?
print(pca.explained_variance_ratio_)
# Output: [0.45, 0.23, 0.12, ...]  # PC1 explains 45% of variance
```

### Other Methods
- **LDA**: Finds dimensions that best separate conditions
- **t-SNE**: Preserves local neighborhood structure
- **ICA**: Finds independent components

---

## Pattern Distinctness Analysis {#pattern-distinctness}

### The Question
"How different are neural patterns between conditions?"

### The Method

1. **Within-condition distances**: How variable is each condition?
   ```python
   # For sooner-smaller trials
   distances_within_SS = pdist(X_sooner_smaller)
   ```

2. **Between-condition distances**: How different are the conditions?
   ```python
   # Between sooner-smaller and larger-later
   distances_between = euclidean_distances(X_SS, X_LL)
   ```

3. **Compare**: Are between-condition distances larger?

### Statistical Test
We use **Cohen's d** (effect size):
```python
d = (mean_between - mean_within) / pooled_standard_deviation
```

**Interpretation**:
- d > 0.8: Large effect (very distinct patterns)
- d > 0.5: Medium effect (moderately distinct)
- d > 0.2: Small effect (somewhat distinct)
- d ≤ 0.2: No effect (patterns not distinct)

### Why This Matters
- High distinctness → Conditions are represented differently
- Low distinctness → Conditions use similar neural patterns
- Tells us about the **selectivity** of neural coding

---

## Representational Similarity Analysis (RSA) {#rsa}

### The Question
"What is the structure of similarity between different stimuli?"

### The Method

1. **Create Representational Similarity Matrix (RSM)**:
   ```python
   # Calculate correlation between every pair of trials
   rsm = np.corrcoef(X)  # X is trials × voxels
   ```

2. **Compare RSMs between conditions**:
   ```python
   rsm_correlation = pearsonr(rsm1.flatten(), rsm2.flatten())
   ```

### What the RSM Shows
- **Diagonal**: Perfect similarity (trial with itself)
- **Off-diagonal**: Similarity between different trials
- **Blocks**: If trials cluster by condition
- **Gradients**: If similarity varies continuously

### Interpretation
- **High RSM correlation**: Similar representational structure across conditions
- **Low RSM correlation**: Different representational structures

### Why RSA is Powerful
- Captures the **full geometry** of representations
- Can compare across brain regions, species, computational models
- Reveals **hierarchical** organization of information

---

## Cross-Validation and Statistical Controls {#cross-validation}

### The Problem: Overfitting
With many features (voxels) and few samples (trials):
- Models can memorize noise
- Results might not generalize
- False confidence in findings

### The Solution: Cross-Validation

**Basic idea**: Train on some data, test on different data

**K-fold cross-validation**:
1. Split data into K parts
2. Train on K-1 parts, test on remaining part
3. Repeat K times
4. Average results

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(classifier, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Additional Controls

1. **Permutation Testing**:
   ```python
   # Shuffle labels many times, see if real result is better than chance
   null_scores = []
   for i in range(1000):
       y_shuffled = shuffle(y)
       null_scores.append(classifier.fit(X, y_shuffled).score(X, y_shuffled))
   
   p_value = (sum(null_scores >= real_score) + 1) / 1001
   ```

2. **Noise Ceiling**:
   - Estimate the best possible performance given noise in data
   - Compare your results to this ceiling

3. **Robustness Checks**:
   - Test with different dimensionality reduction methods
   - Ensure results aren't method-specific

---

## Putting It All Together: Complete Analysis {#complete-analysis}

### Step-by-Step Analysis Pipeline

1. **Load and Preprocess Data**
   ```python
   # Load fMRI data and behavioral data
   fmri_data = load_fmri_data(subject_id)
   behavior = load_behavior_data(subject_id)
   
   # Extract ROI timeseries
   roi_timeseries = extract_roi_data(fmri_data, roi_mask)
   
   # Extract trial-wise patterns
   trial_patterns = extract_trial_patterns(roi_timeseries, behavior)
   ```

2. **Dimensionality Reduction**
   ```python
   pca = PCA(n_components=10)
   X_reduced = pca.fit_transform(trial_patterns)
   ```

3. **Pattern Distinctness**
   ```python
   distinctness = compute_pattern_distinctness(X_reduced, choice_labels)
   ```

4. **RSA Analysis**
   ```python
   rsa_results = compute_rsa(X_reduced, choice_labels)
   ```

5. **Cross-Validated Decoding**
   ```python
   cv_scores = cross_val_score(classifier, X_reduced, choice_labels, cv=5)
   ```

6. **Statistical Testing**
   ```python
   p_value = permutation_test(X_reduced, choice_labels, n_permutations=1000)
   ```

7. **Visualization**
   ```python
   # Plot results, create figures
   visualize_results(X_reduced, choice_labels, value_differences)
   ```

### Interpreting Results

**Good Evidence for Distinct Representations**:
- High pattern distinctness (d > 0.5)
- Above-chance cross-validated decoding (> 50%)
- Significant permutation test (p < 0.05)
- Consistent across different methods

**Evidence for Preserved Geometry**:
- High RSM correlation between conditions
- Similar value coding across conditions
- Results survive robustness checks

### Common Pitfalls to Avoid

1. **Don't skip cross-validation** - Always validate on independent data
2. **Don't ignore multiple comparisons** - Correct for testing multiple ROIs
3. **Don't over-interpret small effects** - Consider effect sizes, not just p-values
4. **Don't forget to visualize** - Plots reveal patterns statistics might miss

---

## Key Takeaways

1. **MVPA reveals information invisible to traditional analysis**
2. **Representational geometry tells us about neural coding principles**
3. **Dimensionality reduction helps focus on meaningful patterns**
4. **Cross-validation and controls are essential for valid conclusions**
5. **Visualization helps interpret complex multivariate results**

## Next Steps

1. **Practice with your own data** - Adapt the code to your specific dataset
2. **Experiment with parameters** - Try different numbers of components, classifiers
3. **Compare across brain regions** - How do different areas represent the same information?
4. **Connect to behavior** - How do neural patterns relate to individual differences?

## Recommended Reading

- Kriegeskorte, N. (2008). Representational similarity analysis
- Haxby, J. V. (2001). Distributed and overlapping representations
- Poldrack, R. A. (2007). Region of interest analysis for fMRI

---

*This tutorial provides a foundation for understanding MVPA and representational geometry. The concepts build on each other, so take time to understand each section before moving to the next!* 