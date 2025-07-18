
import os
from pathlib import Path

# --------------------------
# Main Project Configuration
# --------------------------

# Project structure
class Project:
    ROOT = Path(__file__).parent
    LOG_DIR = ROOT / 'logs'
    CACHE_DIR = ROOT / 'analysis_cache'

# Data paths
class Paths:
    DATA_ROOT = "/oak/stanford/groups/russpold/data/uh2/aim1"
    FMRIPREP_DIR = f"{DATA_ROOT}/derivatives/fmriprep"
    BEHAVIOR_DIR = f"{DATA_ROOT}/behavioral_data/event_files"
    MASKS_DIR = f"{DATA_ROOT}/derivatives/masks"
    OUTPUT_ROOT = f"{DATA_ROOT}/derivatives/mvpa_analysis"
    OUTPUT_DIR = f"{OUTPUT_ROOT}/delay_discounting_results"
    BEHAVIORAL_OUTPUT = f"{OUTPUT_DIR}/behavioral_analysis"
    MVPA_OUTPUT = f"{OUTPUT_DIR}/mvpa_analysis"
    GEOMETRY_OUTPUT = f"{OUTPUT_DIR}/geometry_analysis"

# fMRI parameters
class FMRI:
    TR = 0.68
    SMOOTHING_FWHM = 6.0
    HIGH_PASS_FILTER = 0.01
    STANDARDIZE = True
    DETREND = True
    CONFOUND_STRATEGY = 'auto'

# ROI masks
class ROI:
    CORE_ROIS = ['striatum', 'dlpfc', 'vmpfc']
    OPTIONAL_ROIS = ['left_striatum', 'right_striatum', 'left_dlpfc', 'right_dlpfc', 'acc', 'ofc']
    MASK_FILES = {
        'striatum': f'{Paths.MASKS_DIR}/striatum_mask.nii.gz',
        'dlpfc': f'{Paths.MASKS_DIR}/dlpfc_mask.nii.gz',
        'vmpfc': f'{Paths.MASKS_DIR}/vmpfc_mask.nii.gz',
        'left_striatum': f'{Paths.MASKS_DIR}/left_striatum_mask.nii.gz',
        'right_striatum': f'{Paths.MASKS_DIR}/right_striatum_mask.nii.gz',
        'left_dlpfc': f'{Paths.MASKS_DIR}/left_dlpfc_mask.nii.gz',
        'right_dlpfc': f'{Paths.MASKS_DIR}/right_dlpfc_mask.nii.gz',
        'acc': f'{Paths.MASKS_DIR}/acc_mask.nii.gz',
        'ofc': f'{Paths.MASKS_DIR}/ofc_mask.nii.gz'
    }

# Behavioral analysis
class Behavioral:
    MIN_ACCURACY = 0.6
    MAX_RT = 10.0
    MIN_RT = 0.1
    DISCOUNT_MODEL = 'hyperbolic'
    FIT_METHOD = 'least_squares'
    VARIABLES = [
        'choice', 'choice_binary', 'rt', 'onset', 'delay_days', 
        'amount_small', 'amount_large', 'sv_chosen', 'sv_unchosen', 
        'sv_diff', 'sv_sum', 'later_delay', 'discount_rate', 'model_fit'
    ]

# MVPA analysis
class MVPA:
    CV_FOLDS = 5
    N_PERMUTATIONS = 1000
    RANDOM_STATE = 42
    CLASSIFICATION_TARGETS = ['choice_binary']
    REGRESSION_TARGETS = [
        'sv_diff', 'sv_sum', 'sv_chosen', 'sv_unchosen', 
        'svchosen_unchosen', 'later_delay', 'discount_rate'
    ]
    CLASSIFIER = 'svm'
    REGRESSOR = 'ridge'

# Geometry analysis
class Geometry:
    N_PERMUTATIONS = 1000
    RANDOM_STATE = 42
    ALPHA = 0.05
    N_COMPONENTS_PCA = 15
    N_COMPONENTS_MDS = 8
    STANDARDIZE_DATA = True
    DELAY_SHORT_THRESHOLD = 7
    DELAY_LONG_THRESHOLD = 30
    COMPARISONS = [
        'choice', 'delay_short_vs_long', 'delay_immediate_vs_delayed',
        'sv_chosen_median', 'sv_unchosen_median', 'sv_difference_median'
    ]

# System/execution settings
class System:
    N_JOBS = -1
    LOGGING_LEVEL = 'INFO'
    LOG_FILE = Project.LOG_DIR / 'analysis.log'

# Create directories
def setup_directories():
    Project.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Path(Paths.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(Paths.BEHAVIORAL_OUTPUT).mkdir(parents=True, exist_ok=True)
    Path(Paths.MVPA_OUTPUT).mkdir(parents=True, exist_ok=True)
    Path(Paths.GEOMETRY_OUTPUT).mkdir(parents=True, exist_ok=True)

# Run setup
setup_directories() 