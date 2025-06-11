# MVPA and Representational Geometry Tutorial

Welcome to your comprehensive tutorial for Multi-Voxel Pattern Analysis (MVPA) and representational geometry! This repository contains everything you need to learn these concepts from scratch.

## 🎯 Learning Objectives

By the end of this tutorial, you will understand:
1. **What MVPA is** and why it's more powerful than traditional fMRI analysis
2. **Representational geometry** and how the brain encodes information in patterns
3. **Dimensionality reduction** techniques and when to use them
4. **Statistical validation** methods to ensure robust results
5. **Python programming** for neuroscience research

## 📁 Files in This Repository

### Core Analysis Files
- **`mvpa_delay_discounting.py`** - Complete MVPA pipeline for your real data
- **`mvpa_annotated_example.py`** - Step-by-step tutorial with detailed explanations
- **`mvpa_tutorial.py`** - Interactive demonstrations of key concepts

### Documentation
- **`MVPA_Tutorial_Guide.md`** - Comprehensive conceptual guide
- **`README.md`** - This file
- **`requirements.txt`** - Python packages needed

### HPC Files
- **`submit_mvpa_job.sh`** - SLURM script for running on Stanford's HPC

## 🚀 Getting Started

### Step 1: Set Up Your Environment

First, install the required Python packages:

```bash
# Create a virtual environment (recommended)
python3 -m venv mvpa_env
source mvpa_env/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Start with the Conceptual Guide

Read `MVPA_Tutorial_Guide.md` first. This explains:
- Why MVPA is different from traditional fMRI analysis
- What representational geometry means
- How to interpret your results

### Step 3: Run the Interactive Tutorial

```bash
python3 mvpa_annotated_example.py
```

This will walk you through a complete analysis with synthetic data, explaining each step.

### Step 4: Adapt for Your Real Data

Modify `mvpa_delay_discounting.py` with your actual data paths and run your analysis.

## 🧠 Key Concepts Explained

### What Makes MVPA Different?

**Traditional fMRI (Univariate)**:
```
Voxel 1: Is it more active in condition A vs B?
Voxel 2: Is it more active in condition A vs B?
...
```

**MVPA (Multivariate)**:
```
Pattern across all voxels: What information does it contain?
Can we decode conditions from the pattern?
How do patterns relate to behavior?
```

### Representational Geometry

Think of your brain data as points in a high-dimensional space:
- Each trial is a point
- Each voxel is a dimension
- Similar trials cluster together
- The "geometry" tells us about neural coding

### Your Analysis Pipeline

1. **Load Data** → Extract neural patterns for each trial
2. **Reduce Dimensions** → Use PCA to focus on important patterns
3. **Measure Distinctness** → How different are conditions?
4. **Analyze Similarity** → What's the internal structure?
5. **Test Decoding** → Can we predict behavior from patterns?
6. **Validate Results** → Use cross-validation and controls

## 📊 Understanding Your Results

### Pattern Distinctness
- **Cohen's d > 0.8**: Large effect - very distinct patterns
- **Cohen's d > 0.5**: Medium effect - moderately distinct
- **Cohen's d > 0.2**: Small effect - somewhat distinct

### RSM Correlation
- **r > 0.7**: Very similar representational structure
- **r > 0.3**: Somewhat similar structure
- **r < 0.3**: Different structures

### Decoding Accuracy
- **> 60%**: Good classification
- **55-60%**: Moderate classification
- **< 55%**: Poor classification
- **50%**: Chance level

## 🔬 For Your Delay Discounting Study

Your specific research questions:
1. **Choice Behavior**: Can we decode sooner-smaller vs larger-later choices?
2. **Value Differences**: How do subjective values map onto neural patterns?
3. **Geometry Comparison**: Is the representational structure preserved across choice types?

### Brain Regions of Interest
- **Striatum**: Reward processing and decision-making
- **DLPFC**: Cognitive control and working memory
- **VMPFC**: Value computation and decision-making

### Expected Results
- **Striatum**: Might show strong value coding
- **DLPFC**: Might show choice-type distinctions
- **VMPFC**: Might show preserved geometry across conditions

## 🖥️ Running on Stanford HPC

1. **Upload your data** to OAK storage
2. **Modify paths** in `mvpa_delay_discounting.py`
3. **Submit job**: `sbatch submit_mvpa_job.sh`
4. **Check results** in the generated output files

### HPC Tips
- Use the `--mem=32GB` flag for memory-intensive analyses
- Consider `--array` jobs for multiple subjects
- Save intermediate results to avoid re-computation

## 📈 Interpreting Visualizations

Your analysis will generate several plots:

### PCA Space Plots
- **Points**: Individual trials
- **Colors**: Conditions or value differences
- **Clustering**: Similar trials group together

### Pattern Distance Plots
- **Within-condition**: Variability within each condition
- **Between-condition**: Differences between conditions
- **Higher between-condition**: More distinct representations

### RSM Heatmaps
- **Diagonal**: Perfect similarity (trial with itself)
- **Off-diagonal**: Similarity between different trials
- **Blocks**: Clustering by condition

### Decoding Results
- **Above chance**: Successful classification
- **Error bars**: Reliability across cross-validation folds

## 🚨 Common Pitfalls to Avoid

1. **Don't skip cross-validation** - Always test on independent data
2. **Don't ignore effect sizes** - Statistical significance ≠ practical significance
3. **Don't over-interpret small effects** - Consider the magnitude of differences
4. **Don't forget multiple comparisons** - Correct for testing multiple ROIs
5. **Don't neglect visualization** - Plots reveal patterns statistics might miss

## 🔄 Troubleshooting

### "Low decoding accuracy"
- Check if your conditions are actually different
- Try different numbers of PCA components
- Ensure proper trial extraction and timing

### "No pattern distinctness"
- Verify your ROI masks are correct
- Check for sufficient trials per condition
- Consider different distance metrics

### "Memory errors on HPC"
- Increase memory allocation in SLURM script
- Reduce number of PCA components
- Process subjects individually

## 📚 Next Steps

1. **Run the tutorial** with synthetic data
2. **Adapt for your data** by modifying file paths
3. **Compare across ROIs** to understand regional differences
4. **Relate to behavior** by correlating patterns with individual differences
5. **Test models** by comparing neural patterns to computational predictions

## 📖 Recommended Reading

### Essential Papers
- Haxby et al. (2001) - "Distributed and overlapping representations"
- Kriegeskorte et al. (2008) - "Representational similarity analysis"
- Norman et al. (2006) - "Beyond mind-reading: multi-voxel pattern analysis"

### Advanced Topics
- Poldrack et al. (2009) - "Decoding the large-scale structure of brain function"
- Naselaris et al. (2011) - "Encoding and decoding in fMRI"
- Diedrichsen & Kriegeskorte (2017) - "Representational models"

## 🤝 Getting Help

### If you're stuck:
1. **Check the tutorial guide** - Most concepts are explained there
2. **Look at the annotated code** - Every step is documented
3. **Try with synthetic data first** - Easier to debug
4. **Ask specific questions** - "Why is my decoding accuracy low?" vs "Help!"

### For Stanford students:
- Office hours with your advisor
- CNI user group meetings
- Poldrack lab wiki and resources

---

**Remember**: MVPA is powerful but complex. Take time to understand each concept before moving to the next. The goal is not just to run the analysis, but to understand what it tells you about how the brain works!

Good luck with your research! 🧠✨ 