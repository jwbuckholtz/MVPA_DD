
# fMRI Analysis Pipeline

A refactored and simplified pipeline for analyzing fMRI data, including behavioral modeling, MVPA decoding, and neural geometry analysis.

## Overview

This pipeline provides a streamlined and modular approach to fMRI analysis. The key features are:

- **Unified Configuration**: All settings are in a single `config.py` file.
- **Modular Design**: Core functions are separated into logical modules (`behavioral_modeling.py`, `decoding.py`, `geometry.py`, `data_loader.py`).
- **Simple Entrypoint**: The entire pipeline is run from a single `main.py` script.
- **Clear Structure**: The project is organized for clarity and ease of use.

## New Simplified Structure

The refactored codebase has a simple, modular structure:

- `main.py`: The single entry point to run the entire analysis pipeline.
- `config.py`: A single file for all configuration settings.
- `data_loader.py`: Handles loading of behavioral and fMRI data.
- `behavioral_analysis.py`: Orchestrates the behavioral analysis.
- `behavioral_modeling.py`: Contains the core mathematical models for behavioral analysis.
- `mvpa_analysis.py`: Orchestrates the MVPA.
- `decoding.py`: Contains the core functions for MVPA decoding.
- `geometry_analysis.py`: Orchestrates the neural geometry analysis.
- `geometry.py`: Contains the core functions for neural geometry analysis.

## Usage

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Configure Analysis

Edit the `config.py` file to set data paths, analysis parameters, and other settings.

### 3. Run Analysis

The entire pipeline is run from the `main.py` script. You can run all analyses or select specific ones.

**Run all analyses for default subjects:**
```bash
python main.py
```

**Run specific subjects:**
```bash
python main.py --subjects s001 s004 s005
```

**Skip specific analyses:**
```bash
python main.py --skip-behavioral --skip-geometry
```

### 4. View Results

The analysis scripts will save their results to the output directories specified in `config.py`. The results are saved as `.csv` files for easy analysis.

- **Behavioral Results**: `[OUTPUT_DIR]/behavioral_analysis/behavioral_summary.csv`
- **MVPA Results**: `[OUTPUT_DIR]/mvpa_analysis/mvpa_summary.csv`
- **Geometry Results**: `[OUTPUT_DIR]/geometry_analysis/geometry_summary.csv` 