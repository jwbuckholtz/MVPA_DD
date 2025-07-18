
import nibabel as nib
import pandas as pd
from nilearn.input_data import NiftiMasker
import config

def load_behavioral_data(subject_id):
    """Loads the behavioral data for a single subject."""
    events_file = f"{config.Paths.BEHAVIOR_DIR}/{subject_id}_discountFix_events.tsv"
    return pd.read_csv(events_file, sep='\t')

def load_fmri_data(subject_id):
    """Loads the fMRI data for a single subject."""
    fmri_file = f"{config.Paths.FMRIPREP_DIR}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-discountFix_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    return nib.load(fmri_file)

def extract_roi_data(fmri_data, roi_mask_file, confounds=None):
    """Extracts voxel data from an ROI using a NiftiMasker."""
    masker = NiftiMasker(mask_img=roi_mask_file, standardize=True, detrend=True, high_pass=config.FMRI.HIGH_PASS_FILTER, t_r=config.FMRI.TR)
    return masker.fit_transform(fmri_data, confounds=confounds) 