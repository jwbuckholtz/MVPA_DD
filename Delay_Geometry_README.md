# Delay Geometry Analysis for MVPA

This module provides comprehensive tools for analyzing how the geometric properties of neural representations change across different delay conditions in delay discounting tasks.

## 🎯 Research Questions

These scripts help answer key questions about temporal decision-making:

1. **How does representational geometry change with delay length?**
   - Do neural patterns become more/less distinct as delays increase?
   - Is there a systematic transformation in neural space?

2. **Are there delay-dependent geometric transformations?**
   - Can we predict one delay condition from another?
   - What is the "trajectory" of representations across delays?

3. **Which brain regions show delay-dependent changes?**
   - Do striatum, DLPFC, and VMPFC show different patterns?
   - Are some regions more sensitive to delay than others?

4. **How do value representations evolve across delays?**
   - Does the value gradient change with delay?
   - Are value representations preserved or transformed?

## 📁 Files Overview

### Core Analysis Scripts

- **`delay_geometry_analysis.py`** - Main analysis pipeline
- **`geometric_transformation_analysis.py`** - Advanced transformation methods
- **`mvpa_delay_discounting.py`** - Base MVPA pipeline (from previous tutorial)

### Key Functions

#### `delay_geometry_analysis.py`
- `load_delay_specific_data()` - Organize data by delay condition
- `compute_geometric_properties()` - Calculate comprehensive geometry metrics
- `analyze_delay_progression()` - Track changes across delays
- `visualize_delay_geometry()` - Create comprehensive visualizations

#### `geometric_transformation_analysis.py`
- `compute_manifold_alignment()` - Align neural manifolds between delays
- `compute_geodesic_distances()` - Measure distances on neural manifolds
- `analyze_trajectory_dynamics()` - Study representational trajectories
- `compute_information_geometry_metrics()` - Information-theoretic measures

## 🔬 Geometric Properties Measured

### 1. **Centroid Distance**
```python
# Distance between choice-type centroids
centroid_ss = np.mean(X_reduced[labels == 0], axis=0)
centroid_ll = np.mean(X_reduced[labels == 1], axis=0)
distance = np.linalg.norm(centroid_ss - centroid_ll)
```
**Interpretation**: How far apart are sooner-smaller vs larger-later representations?

### 2. **Separability Index**
```python
# Ratio of between-condition to within-condition distances
separability = between_distance / within_distance
```
**Interpretation**: How well can we distinguish the two choice types?

### 3. **Value Gradient Strength**
```python
# Correlation between neural position and subjective value
value_gradient = np.sqrt(corr_pc1**2 + corr_pc2**2)
```
**Interpretation**: How strongly is value encoded in the neural geometry?

### 4. **Effective Dimensionality**
```python
# Participation ratio of eigenvalues
dimensionality = (sum(eigenvals)**2) / sum(eigenvals**2)
```
**Interpretation**: How many dimensions are actively used for representation?

### 5. **Manifold Curvature**
```python
# Local curvature based on PCA in neighborhoods
curvature = 1 - explained_variance_ratio[0]
```
**Interpretation**: How "curved" or "flat" is the neural manifold?

### 6. **Transformation Quality**
```python
# Procrustes alignment between delay conditions
alignment_quality = 1 - procrustes_disparity
```
**Interpretation**: How well can one delay condition be transformed to another?

## 🚀 Usage Guide

### Step 1: Prepare Your Data

Ensure your data is organized with delay information:

```python
# Your behavioral data should include:
behavior_data = pd.DataFrame({
    'trial': [1, 2, 3, ...],
    'delay': [1, 7, 30, ...],  # Delay in days
    'choice': [0, 1, 0, ...],  # 0=SS, 1=LL
    'chosen_value': [10, 15, 8, ...],
    'unchosen_value': [12, 10, 15, ...]
})
```

### Step 2: Run Basic Delay Geometry Analysis

```python
from delay_geometry_analysis import main_delay_geometry_analysis

# Analyze first 10 subjects across all ROIs
subject_ids = [f"{i:03d}" for i in range(1, 11)]
main_delay_geometry_analysis(subject_ids)
```

### Step 3: Run Advanced Geometric Analysis

```python
from geometric_transformation_analysis import main_geometric_analysis

# Advanced transformation analysis
main_geometric_analysis(subject_ids)
```

### Step 4: Interpret Results

Check the generated visualizations and summary statistics.

## 📊 Understanding Your Results

### Visualization Outputs

#### 1. **Delay Progression Plots**
- **PCA embeddings** for each delay condition
- **Property trajectories** showing how metrics change with delay
- **Transformation heatmaps** showing alignment quality between delays

#### 2. **3D Trajectory Plots**
- **Centroid trajectories** through neural space
- **Velocity profiles** showing rate of change
- **Curvature evolution** across delays

#### 3. **Information Geometry Plots**
- **Jensen-Shannon divergence** between consecutive delays
- **Alignment quality matrices** across all delay pairs
- **Dimensionality evolution** with delay

### Key Metrics to Examine

#### **Delay Sensitivity Score**
```python
# Average absolute correlation between delay and geometric properties
delay_sensitivity = mean([abs(corr) for corr in trend_correlations])
```

**Interpretation**:
- **> 0.3**: High delay sensitivity - geometry changes significantly
- **0.15-0.3**: Moderate delay sensitivity - some changes
- **< 0.15**: Low delay sensitivity - stable across delays

#### **Transformation Consistency**
```python
# How consistently can delays be aligned with each other
consistency = 1 - std(transformation_qualities)
```

**Interpretation**:
- **> 0.8**: Highly consistent transformations
- **0.5-0.8**: Moderately consistent
- **< 0.5**: Inconsistent transformations

#### **Trajectory Smoothness**
```python
# How smooth is the trajectory through neural space
smoothness = 1 / (1 + var(accelerations))
```

**Interpretation**:
- **> 0.7**: Smooth, systematic changes
- **0.3-0.7**: Moderately systematic
- **< 0.3**: Erratic, non-systematic changes

## 🧠 Expected Results by Brain Region

### **Striatum**
- **High delay sensitivity** - reward regions should be sensitive to delay
- **Strong value gradients** - should encode subjective value consistently
- **Moderate transformation quality** - some systematic changes expected

### **DLPFC**
- **Moderate delay sensitivity** - cognitive control may adapt to delay
- **High separability** - should distinguish choice types well
- **High transformation consistency** - systematic cognitive processes

### **VMPFC**
- **Variable delay sensitivity** - depends on individual differences
- **Strong value gradients** - key region for value computation
- **High dimensionality** - complex value representations

## 🔧 Customization Options

### Modify Geometric Properties

Add your own geometric measures:

```python
def compute_custom_property(X_reduced, labels, values):
    # Your custom geometric property
    custom_metric = your_calculation(X_reduced)
    return custom_metric

# Add to the properties dictionary in compute_geometric_properties()
```

### Change Dimensionality Reduction

Use different methods:

```python
# Instead of PCA
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import FastICA

# t-SNE for non-linear embedding
tsne = TSNE(n_components=2)
X_reduced = tsne.fit_transform(patterns)

# ICA for independent components
ica = FastICA(n_components=10)
X_reduced = ica.fit_transform(patterns)
```

### Adjust Analysis Parameters

```python
# In delay_geometry_analysis.py
pca = PCA(n_components=15)  # More/fewer components
curvature = compute_manifold_curvature(patterns, k=10)  # More neighbors
```

## 📈 Statistical Considerations

### Multiple Comparisons
When testing multiple ROIs and delay conditions:

```python
from statsmodels.stats.multitest import multipletests

# Correct p-values for multiple comparisons
_, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

### Effect Size Interpretation
Use Cohen's conventions for effect sizes:

- **Small effect**: d = 0.2, r = 0.1
- **Medium effect**: d = 0.5, r = 0.3  
- **Large effect**: d = 0.8, r = 0.5

### Sample Size Considerations
- **Minimum trials per delay**: 20-30 for stable estimates
- **Minimum subjects**: 15-20 for group-level inference
- **Cross-validation**: Always use for any predictive analyses

## 🚨 Common Issues and Solutions

### **Issue**: Low transformation quality across all delays
**Solution**: 
- Check if delays actually differ behaviorally
- Verify ROI extraction is correct
- Try different alignment methods (CCA vs Procrustes)

### **Issue**: No systematic delay trends
**Solution**:
- Examine individual subject patterns
- Check for sufficient trials per delay condition
- Consider non-linear trends (quadratic, exponential)

### **Issue**: High variability across subjects
**Solution**:
- Use mixed-effects models for group analysis
- Examine individual difference predictors
- Consider hierarchical clustering of subjects

### **Issue**: Memory errors with large datasets
**Solution**:
- Process subjects individually
- Reduce number of PCA components
- Use batch processing for large subject groups

## 🔄 Integration with Main MVPA Pipeline

These scripts integrate with your main MVPA analysis:

```python
# 1. Run basic MVPA first
from mvpa_delay_discounting import main_analysis
main_analysis()

# 2. Then run delay-specific geometry analysis
from delay_geometry_analysis import main_delay_geometry_analysis
main_delay_geometry_analysis(subject_ids)

# 3. Finally, advanced geometric transformations
from geometric_transformation_analysis import main_geometric_analysis
main_geometric_analysis(subject_ids)
```

## 📚 Theoretical Background

### **Representational Geometry**
The idea that neural populations create geometric spaces where:
- Similar stimuli are close together
- Different stimuli are far apart
- The geometry reveals computational principles

### **Delay Discounting Theory**
As delays increase:
- **Hyperbolic discounting**: Value decreases hyperbolically
- **Neural adaptation**: Representations may adapt to delay context
- **Individual differences**: People vary in delay sensitivity

### **Manifold Learning**
Neural data lies on lower-dimensional manifolds:
- **Intrinsic dimensionality** < number of neurons/voxels
- **Geometric properties** reveal computational structure
- **Transformations** show how representations change

## 📖 Recommended Reading

### Core Papers
- **Kriegeskorte & Kievit (2013)** - Representational geometry
- **Gallego et al. (2017)** - Neural manifolds in motor cortex
- **Bernardi et al. (2020)** - Geometry of neural representations

### Delay Discounting
- **Kable & Glimcher (2007)** - Neural basis of delay discounting
- **Peters & Büchel (2011)** - Episodic future thinking and delay discounting
- **Lempert et al. (2019)** - Neural and computational mechanisms

### Methods
- **Cunningham & Yu (2014)** - Dimensionality reduction for neural data
- **Williams et al. (2018)** - Unsupervised discovery of demixed representations
- **Gao & Ganguli (2015)** - On simplicity and complexity in neural representations

---

**Remember**: Geometric analysis reveals the computational principles underlying neural representations. Take time to understand what each metric means for your specific research questions!

Good luck with your delay discounting geometry analysis! 🧠⏰ 