# Delay Discounting Neural Geometry Analysis

This script provides specialized neural geometry analysis for delay discounting fMRI experiments. It performs comprehensive comparisons between different experimental conditions using representational similarity analysis (RSA), dimensionality reduction, and statistical testing.

## Features

### **Supported Comparison Types**

1. **Choice Comparisons**: Sooner-Smaller vs Larger-Later choices
2. **Delay Comparisons**: 
   - Short vs Long delays (customizable thresholds)
   - Immediate vs Delayed trials
   - Median split of delay lengths
3. **Subjective Value Comparisons**:
   - High vs Low chosen option value
   - High vs Low unchosen option value  
   - High vs Low difference in subjective values
4. **Value Difference Comparisons**: Similar vs Different option values (difficulty)

### **Analysis Methods**

#### **Standard Geometric Analyses**
- **Representational Similarity Analysis (RSA)**
- **Principal Component Analysis (PCA)**
- **Multidimensional Scaling (MDS)**
- **Distance analysis** with permutation testing
- **Dimensionality analysis** (intrinsic dimensionality, participation ratio)

#### **Advanced Geometric Analyses**
- **Manifold Alignment**: Procrustes analysis and Canonical Correlation Analysis (CCA)
- **Geodesic Distance Analysis**: Non-linear distance computation using Isomap
- **Manifold Curvature Estimation**: Local curvature analysis of neural manifolds
- **Information Geometry Metrics**: KL divergence, Jensen-Shannon divergence, Wasserstein distance
- **Trajectory Dynamics**: Analysis of how neural representations change across conditions

## Input Data Requirements

### **Neural Data**
- **Format**: `.npy` or `.csv` files
- **Shape**: `(n_trials, n_features)` where:
  - `n_trials` = number of experimental trials
  - `n_features` = number of voxels/ROI features

### **Behavioral Data**
- **Format**: `.csv` file with the following columns:

| Column | Description | Values |
|--------|-------------|---------|
| `choice` | Participant's choice | 0 = Sooner-Smaller, 1 = Larger-Later |
| `delay_days` | Delay to larger reward | Numeric (days) |
| `chosen_sv` | Subjective value of chosen option | Numeric (0-1) |
| `unchosen_sv` | Subjective value of unchosen option | Numeric (0-1) |

## Usage Examples

### **1. Basic Analysis with All Comparisons**
```bash
python3 delay_discounting_geometry_analysis.py \
    --neural-data roi_timeseries.npy \
    --behavioral-data trial_data.csv \
    --roi-name "DLPFC" \
    --config dd_geometry_config.json
```

### **2. Specific Comparisons Only**
```bash
python3 delay_discounting_geometry_analysis.py \
    --neural-data roi_data.npy \
    --behavioral-data behavior.csv \
    --roi-name "Striatum" \
    --comparisons choice delay_immediate_vs_delayed sv_chosen_median
```

### **3. Custom Output Directory**
```bash
python3 delay_discounting_geometry_analysis.py \
    --neural-data neural_data.npy \
    --behavioral-data behavioral_data.csv \
    --roi-name "VMPFC" \
    --output-dir ./vmpfc_results/
```

### **4. Test with Example Data**
```bash
python3 delay_discounting_geometry_analysis.py --example --roi-name "Test_ROI"
```

## Configuration

### **Configuration File (JSON)**
```json
{
  "output_dir": "./dd_geometry_results",
  "n_permutations": 1000,
  "random_state": 42,
  "alpha": 0.05,
  "n_components_pca": 15,
  "n_components_mds": 8,
  "standardize_data": true,
  "plot_format": "png",
  "dpi": 300,
  "delay_short_threshold": 7,
  "delay_long_threshold": 30,
  "advanced_geometry": {
    "enable_manifold_alignment": true,
    "enable_geodesic_analysis": true,
    "enable_curvature_analysis": true,
    "enable_information_geometry": true,
    "isomap_n_neighbors": 5,
    "curvature_n_neighbors": 5,
    "kde_bandwidth": "auto"
  }
}
```

### **Key Parameters**

#### **Basic Parameters**
- `delay_short_threshold`: Days ≤ this are considered "short" delays (default: 7)
- `delay_long_threshold`: Days ≥ this are considered "long" delays (default: 30)
- `n_permutations`: Number of permutations for statistical testing (default: 1000)
- `standardize_data`: Whether to z-score neural data (default: true)

#### **Advanced Geometry Parameters**
- `enable_manifold_alignment`: Enable Procrustes and CCA alignment analysis (default: true)
- `enable_geodesic_analysis`: Enable geodesic distance computation (default: true)
- `enable_curvature_analysis`: Enable local curvature estimation (default: true)
- `enable_information_geometry`: Enable information-theoretic metrics (default: true)
- `isomap_n_neighbors`: Number of neighbors for Isomap geodesic computation (default: 5)
- `curvature_n_neighbors`: Number of neighbors for curvature estimation (default: 5)
- `kde_bandwidth`: Bandwidth for kernel density estimation ('auto' or numeric)

## Available Comparisons

| Comparison | Description | Behavioral Columns Required |
|------------|-------------|------------------------------|
| `choice` | Sooner-Smaller vs Larger-Later | `choice` |
| `delay_short_vs_long` | Short vs Long delays | `delay_days` |
| `delay_immediate_vs_delayed` | Immediate vs Delayed | `delay_days` |
| `sv_chosen_median` | High vs Low chosen SV | `chosen_sv` |
| `sv_unchosen_median` | High vs Low unchosen SV | `unchosen_sv` |
| `sv_difference_median` | High vs Low SV difference | `chosen_sv`, `unchosen_sv` |
| `value_diff_terciles` | Similar vs Different values | `chosen_sv`, `unchosen_sv` |

## Output Files

### **Results Files**
- `{ROI_NAME}_{comparison}_results.json`: Detailed results for each comparison
- `{ROI_NAME}_summary_report.txt`: Comprehensive text summary
- `visualizations/{ROI_NAME}_{comparison}_advanced_geometry.png`: Advanced geometry visualizations

### **Results Structure**
Each comparison result now includes:

#### **Standard Results**
- RSA correlation and significance
- PCA/MDS separation ratios and p-values
- Dimensionality analysis results
- Statistical interpretations

#### **Advanced Geometry Results**
- **Manifold Alignment**:
  - Procrustes alignment quality and transformation matrices
  - Canonical correlation coefficients
- **Information Geometry**:
  - KL divergence between condition distributions
  - Jensen-Shannon divergence
  - Wasserstein distance approximation
- **Geodesic Analysis**:
  - Geodesic distance matrices for each condition
  - Mean geodesic distances and variance
- **Curvature Analysis**:
  - Local curvature estimates for each data point
  - Mean curvature by condition

### **Visualization Outputs**
The script automatically generates comprehensive visualizations including:
1. **Manifold Alignment Quality**: Comparison of Procrustes vs CCA alignment
2. **Information Geometry Metrics**: Bar charts of divergence measures
3. **Curvature Analysis**: Mean curvature by condition
4. **Geodesic Distance Comparison**: Condition-wise geodesic distances
5. **PCA Embedding Plots**: 2D scatter plots with condition coloring
6. **Distance Analysis Summary**: Within vs between condition distances

## Example Workflow

### **1. Prepare Your Data**
```python
# Neural data: Extract ROI time series or beta estimates
# Shape: (n_trials, n_voxels)
neural_data = extract_roi_timeseries(roi_mask, preprocessed_data)
np.save('dlpfc_timeseries.npy', neural_data)

# Behavioral data: Trial-by-trial information
behavioral_df = pd.DataFrame({
    'choice': [0, 1, 1, 0, ...],  # 0=SS, 1=LL
    'delay_days': [0, 7, 30, 1, ...],
    'chosen_sv': [0.4, 0.8, 0.6, 0.5, ...],
    'unchosen_sv': [0.3, 0.7, 0.4, 0.6, ...]
})
behavioral_df.to_csv('trial_data.csv', index=False)
```

### **2. Run Analysis**
```bash
python3 delay_discounting_geometry_analysis.py \
    --neural-data dlpfc_timeseries.npy \
    --behavioral-data trial_data.csv \
    --roi-name "DLPFC" \
    --config dd_geometry_config.json
```

### **3. Interpret Results**
The script will output:
- Statistical significance of geometric differences
- Effect sizes (separation ratios)
- Interpretations of each comparison
- Comprehensive summary report
- **Advanced geometry visualizations** showing manifold properties and transformations

## Research Applications

### **Typical Delay Discounting Questions**
1. **Do sooner-smaller and larger-later choices have different neural geometries?**
   - Use: `choice` comparison
   - **Advanced insights**: Manifold alignment quality, information divergence

2. **How does delay length affect neural representations?**
   - Use: `delay_short_vs_long`, `delay_immediate_vs_delayed`
   - **Advanced insights**: Curvature differences, geodesic trajectory analysis

3. **Are high-value and low-value trials represented differently?**
   - Use: `sv_chosen_median`, `sv_unchosen_median`
   - **Advanced insights**: Information geometry metrics, manifold topology

4. **Do easy vs difficult choices have different neural geometries?**
   - Use: `value_diff_terciles` (similar values = harder choices)
   - **Advanced insights**: Local curvature, geodesic distance patterns

### **Advanced Geometric Questions**
5. **How similar are the neural manifolds between conditions?**
   - **Procrustes alignment**: Measures how well one manifold can be transformed to match another
   - **CCA alignment**: Finds maximally correlated dimensions between conditions

6. **Do conditions differ in their intrinsic manifold properties?**
   - **Curvature analysis**: Higher curvature indicates more complex local geometry
   - **Geodesic distances**: Non-linear distances that respect manifold structure

7. **How much information distinguishes between conditions?**
   - **KL/JS divergence**: Information-theoretic measures of condition separability
   - **Wasserstein distance**: Geometric distance between probability distributions

### **Multi-ROI Analysis**
```bash
# Run for multiple ROIs
for roi in "DLPFC" "Striatum" "VMPFC" "ACC"; do
    python3 delay_discounting_geometry_analysis.py \
        --neural-data ${roi}_timeseries.npy \
        --behavioral-data trial_data.csv \
        --roi-name $roi \
        --output-dir ./results_${roi}/
done
```

## Statistical Interpretation

### **Standard Metrics**

#### **RSA Correlation**
- **Positive**: Trials from same condition are more similar
- **Negative**: Trials from different conditions are more similar
- **Magnitude**: Effect size (weak < 0.3 < moderate < 0.5 < strong)

#### **Separation Ratio**
- **> 1.0**: Between-condition distances > within-condition distances
- **Higher values**: Better separated conditions in neural space
- **P-value**: Statistical significance via permutation testing

#### **Dimensionality Analysis**
- **Participation Ratio**: Effective dimensionality of neural representations
- **Intrinsic Dimensionality**: Components needed for 80%/90% variance

### **Advanced Geometry Metrics**

#### **Manifold Alignment**
- **Procrustes Alignment Quality**: 0-1 scale (1 = perfect alignment)
  - **> 0.8**: Very similar manifold shapes
  - **0.5-0.8**: Moderately similar shapes
  - **< 0.5**: Different manifold geometries
- **CCA Correlations**: Maximum correlations between condition subspaces
  - **> 0.7**: Strong shared structure
  - **0.3-0.7**: Moderate shared structure
  - **< 0.3**: Little shared structure

#### **Information Geometry**
- **KL Divergence**: Information required to distinguish conditions
  - **Higher values**: More distinguishable conditions
  - **Range**: 0 to ∞ (logarithmic scale)
- **Jensen-Shannon Divergence**: Symmetric information distance
  - **Range**: 0 to ln(2) ≈ 0.693
  - **> 0.5**: Highly distinguishable conditions
- **Wasserstein Distance**: Geometric cost of transforming one distribution to another
  - **Higher values**: Greater geometric separation

#### **Manifold Properties**
- **Curvature**: Local geometric complexity
  - **Higher values**: More curved/complex local geometry
  - **Range**: 0 to 1
- **Geodesic Distances**: Non-linear manifold distances
  - **Compare to Euclidean**: Higher ratio indicates more curved manifold
  - **Condition differences**: Reveal non-linear geometric properties

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

**Note**: The advanced geometry features require:
- `scikit-learn >= 0.24.0` (for Isomap and CCA)
- `scipy >= 1.7.0` (for advanced statistical functions)
- `matplotlib >= 3.3.0` (for 3D visualizations)

## Troubleshooting

### **Common Issues**
1. **Data Shape Mismatch**: Ensure neural and behavioral data have same number of trials
2. **Missing Columns**: Check behavioral data has required columns for chosen comparisons
3. **Small Sample Sizes**: Some comparisons may exclude trials (e.g., tercile splits)

### **Performance Tips**
- Use fewer permutations for faster testing (`n_permutations: 500`)
- Reduce PCA/MDS components for large datasets
- Run specific comparisons only (not all) for faster analysis

#### **Advanced Geometry Performance**
- **Large datasets (>1000 trials)**: Consider reducing `isomap_n_neighbors` and `curvature_n_neighbors`
- **High-dimensional data (>500 features)**: PCA preprocessing is automatically applied for efficiency
- **Memory considerations**: Information geometry uses 2D KDE approximations for computational tractability
- **Disable specific analyses**: Set advanced geometry flags to `false` in config for faster computation

#### **Computational Complexity**
- **Standard analyses**: O(n²) for distance computations
- **Geodesic analysis**: O(n² log n) for Isomap
- **Curvature analysis**: O(nk²) where k is number of neighbors
- **Information geometry**: O(n) for KDE estimation on 2D projections

## Integration with Main Pipeline

This script is designed to work with the main `delay_discounting_mvpa_pipeline.py`:

```python
# Extract neural data from main pipeline
roi_data = roi_analysis.extract_roi_data(roi_mask)
np.save('roi_neural_data.npy', roi_data)

# Use existing behavioral data
behavioral_data.to_csv('behavioral_data.csv', index=False)

# Run geometry analysis
!python3 delay_discounting_geometry_analysis.py \
    --neural-data roi_neural_data.npy \
    --behavioral-data behavioral_data.csv \
    --roi-name "Custom_ROI"
``` 