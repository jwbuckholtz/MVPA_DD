#!/usr/bin/env python3
"""
Validate ROI Masks for MVPA Analysis Pipeline
============================================

This script validates pre-existing ROI masks stored on OAK for the delay discounting
fMRI analysis pipeline. It checks mask availability, integrity, and creates 
visualizations for quality control.

ROI Masks Expected:
- Core masks: Striatum, DLPFC, VMPFC
- Optional masks: Left/Right Striatum, Left/Right DLPFC, ACC, OFC

Author: Cognitive Neuroscience Lab, Stanford University
"""

# Import logger utilities for standardized setup
from logger_utils import (
    setup_pipeline_environment, create_analysis_parser, 
    setup_script_logging
)

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting, image
import pandas as pd

# Data utilities
from data_utils import check_mask_exists, load_mask, DataError
from oak_storage_config import OAKConfig


class MaskValidator:
    """Class for validating ROI masks"""
    
    def __init__(self, config: OAKConfig = None):
        self.config = config or OAKConfig()
        self.validation_results = {}
        self.core_masks_valid = False
        self.optional_masks_available = []
        
    def validate_single_mask(self, roi_name: str, mask_path: str) -> dict:
        """
        Validate a single ROI mask
        
        Parameters:
        -----------
        roi_name : str
            Name of the ROI
        mask_path : str
            Path to mask file
            
        Returns:
        --------
        dict : Validation results
        """
        result = {
            'roi_name': roi_name,
            'mask_path': mask_path,
            'exists': False,
            'valid': False,
            'n_voxels': 0,
            'resolution': None,
            'dimensions': None,
            'error': None
        }
        
        try:
            # Check if file exists
            if not check_mask_exists(mask_path):
                result['error'] = 'File not found'
                return result
            
            result['exists'] = True
            
            # Load and validate mask
            mask_img = load_mask(mask_path, validate=True)
            mask_data = mask_img.get_fdata()
            
            # Basic validation
            result['n_voxels'] = int(np.sum(mask_data > 0))
            result['dimensions'] = mask_data.shape
            result['resolution'] = mask_img.header.get_zooms()[:3]
            
            # Quality checks
            if result['n_voxels'] == 0:
                result['error'] = 'Empty mask (no voxels)'
                return result
                
            if result['n_voxels'] < 10:
                result['error'] = f'Very small mask ({result["n_voxels"]} voxels)'
                return result
                
            # Check for reasonable mask size
            if result['n_voxels'] > 50000:
                result['error'] = f'Unusually large mask ({result["n_voxels"]} voxels)'
                return result
            
            # Check data type and range
            unique_values = np.unique(mask_data)
            if not (np.all(unique_values >= 0) and np.all(unique_values <= 1)):
                result['error'] = f'Invalid mask values: {unique_values}'
                return result
            
            result['valid'] = True
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def validate_all_masks(self) -> pd.DataFrame:
        """
        Validate all ROI masks
        
        Returns:
        --------
        pd.DataFrame : Validation results for all masks
        """
        print("Validating ROI masks...")
        
        validation_data = []
        
        # Validate core masks
        print("\nCore ROI masks:")
        core_valid_count = 0
        
        for roi_name in self.config.CORE_ROI_MASKS:
            if roi_name in self.config.ROI_MASKS:
                mask_path = self.config.ROI_MASKS[roi_name]
                result = self.validate_single_mask(roi_name, mask_path)
                result['mask_type'] = 'core'
                
                # Print status
                if result['valid']:
                    print(f"  ✓ {roi_name}: {result['n_voxels']} voxels")
                    core_valid_count += 1
                else:
                    print(f"  ✗ {roi_name}: {result['error']}")
                
                validation_data.append(result)
            else:
                print(f"  ✗ {roi_name}: Not configured")
        
        self.core_masks_valid = (core_valid_count == len(self.config.CORE_ROI_MASKS))
        
        # Validate optional masks
        print("\nOptional ROI masks:")
        optional_valid_count = 0
        
        for roi_name in self.config.OPTIONAL_ROI_MASKS:
            if roi_name in self.config.ROI_MASKS:
                mask_path = self.config.ROI_MASKS[roi_name]
                result = self.validate_single_mask(roi_name, mask_path)
                result['mask_type'] = 'optional'
                
                # Print status
                if result['valid']:
                    print(f"  ✓ {roi_name}: {result['n_voxels']} voxels")
                    self.optional_masks_available.append(roi_name)
                    optional_valid_count += 1
                else:
                    print(f"  ✗ {roi_name}: {result['error']}")
                
                validation_data.append(result)
            else:
                print(f"  - {roi_name}: Not configured")
        
        # Create results DataFrame
        results_df = pd.DataFrame(validation_data)
        
        # Summary
        print(f"\nValidation Summary:")
        print(f"Core masks valid: {core_valid_count}/{len(self.config.CORE_ROI_MASKS)}")
        print(f"Optional masks available: {optional_valid_count}/{len(self.config.OPTIONAL_ROI_MASKS)}")
        
        return results_df
    
    def create_mask_visualizations(self, results_df: pd.DataFrame, output_dir: str = None):
        """
        Create visualizations of valid masks
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Validation results
        output_dir : str, optional
            Output directory for visualizations
        """
        if output_dir is None:
            output_dir = Path("./mask_validation_outputs")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter valid masks
        valid_masks = results_df[results_df['valid'] == True]
        
        if len(valid_masks) == 0:
            print("No valid masks to visualize")
            return
        
        print(f"\nCreating visualizations for {len(valid_masks)} valid masks...")
        
        # Create grid visualization
        n_masks = len(valid_masks)
        n_cols = min(3, n_masks)
        n_rows = (n_masks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_masks == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each valid mask
        for i, (_, mask_info) in enumerate(valid_masks.iterrows()):
            try:
                plotting.plot_roi(
                    mask_info['mask_path'],
                    title=f"{mask_info['roi_name']}\n({mask_info['n_voxels']} voxels)",
                    axes=axes[i] if n_masks > 1 else axes[0],
                    cmap='Reds',
                    alpha=0.8
                )
            except Exception as e:
                print(f"Failed to plot {mask_info['roi_name']}: {e}")
        
        # Remove empty subplots
        for i in range(len(valid_masks), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        viz_file = output_dir / 'roi_masks_validation.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mask visualization saved: {viz_file}")
        
        # Create summary table
        summary_file = output_dir / 'mask_validation_summary.csv'
        results_df.to_csv(summary_file, index=False)
        print(f"Validation summary saved: {summary_file}")
    
    def create_detailed_report(self, results_df: pd.DataFrame, output_dir: str = None):
        """
        Create detailed validation report
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Validation results
        output_dir : str, optional
            Output directory for report
        """
        if output_dir is None:
            output_dir = Path("./mask_validation_outputs")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / 'mask_validation_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("ROI Mask Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration: {self.config.__class__.__name__}\n")
            f.write(f"Masks directory: {self.config.MASKS_DIR}\n\n")
            
            # Core masks section
            f.write("CORE ROI MASKS\n")
            f.write("-" * 20 + "\n")
            
            core_masks = results_df[results_df['mask_type'] == 'core']
            for _, mask in core_masks.iterrows():
                status = "✓ VALID" if mask['valid'] else "✗ INVALID"
                f.write(f"{mask['roi_name']}: {status}\n")
                if mask['valid']:
                    f.write(f"  - Voxels: {mask['n_voxels']}\n")
                    f.write(f"  - Dimensions: {mask['dimensions']}\n")
                    f.write(f"  - Resolution: {mask['resolution']}\n")
                else:
                    f.write(f"  - Error: {mask['error']}\n")
                f.write(f"  - Path: {mask['mask_path']}\n\n")
            
            # Optional masks section
            f.write("OPTIONAL ROI MASKS\n")
            f.write("-" * 20 + "\n")
            
            optional_masks = results_df[results_df['mask_type'] == 'optional']
            for _, mask in optional_masks.iterrows():
                status = "✓ AVAILABLE" if mask['valid'] else "✗ UNAVAILABLE"
                f.write(f"{mask['roi_name']}: {status}\n")
                if mask['valid']:
                    f.write(f"  - Voxels: {mask['n_voxels']}\n")
                    f.write(f"  - Dimensions: {mask['dimensions']}\n")
                    f.write(f"  - Resolution: {mask['resolution']}\n")
                else:
                    f.write(f"  - Error: {mask['error']}\n")
                f.write(f"  - Path: {mask['mask_path']}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            valid_core = core_masks['valid'].sum()
            total_core = len(core_masks)
            valid_optional = optional_masks['valid'].sum()
            total_optional = len(optional_masks)
            
            f.write(f"Core masks valid: {valid_core}/{total_core}\n")
            f.write(f"Optional masks available: {valid_optional}/{total_optional}\n")
            f.write(f"Pipeline ready: {'YES' if self.core_masks_valid else 'NO'}\n\n")
            
            if not self.core_masks_valid:
                f.write("REQUIRED ACTIONS:\n")
                f.write("- Ensure all core ROI masks are available on OAK\n")
                f.write("- Check mask file paths and permissions\n")
                f.write("- Contact data administrator if masks are missing\n")
        
        print(f"Detailed report saved: {report_file}")
    
    def get_available_rois(self) -> list:
        """
        Get list of available (valid) ROI names
        
        Returns:
        --------
        list : Available ROI names
        """
        available_rois = []
        
        for roi_name, mask_path in self.config.ROI_MASKS.items():
            if check_mask_exists(mask_path):
                try:
                    load_mask(mask_path, validate=True)
                    available_rois.append(roi_name)
                except:
                    continue
        
        return available_rois


def check_oak_connectivity():
    """Check if OAK storage is accessible"""
    config = OAKConfig()
    
    print("Checking OAK connectivity...")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Masks directory: {config.MASKS_DIR}")
    
    if Path(config.DATA_ROOT).exists():
        print("✓ OAK data root accessible")
    else:
        print("✗ OAK data root not accessible")
        return False
    
    if Path(config.MASKS_DIR).exists():
        print("✓ OAK masks directory accessible")
        return True
    else:
        print("✗ OAK masks directory not accessible")
        return False


def create_mask_inventory(config: OAKConfig = None) -> pd.DataFrame:
    """
    Create inventory of all mask files in the masks directory
    
    Parameters:
    -----------
    config : OAKConfig, optional
        Configuration object
        
    Returns:
    --------
    pd.DataFrame : Inventory of mask files
    """
    if config is None:
        config = OAKConfig()
    
    masks_dir = Path(config.MASKS_DIR)
    
    if not masks_dir.exists():
        print(f"Masks directory not found: {masks_dir}")
        return pd.DataFrame()
    
    print(f"Scanning masks directory: {masks_dir}")
    
    # Find all .nii.gz files
    mask_files = list(masks_dir.glob("*.nii.gz"))
    
    inventory_data = []
    
    for mask_file in mask_files:
        try:
            # Get file info
            file_info = {
                'filename': mask_file.name,
                'full_path': str(mask_file),
                'size_mb': mask_file.stat().st_size / (1024 * 1024),
                'configured': False,
                'roi_name': None
            }
            
            # Check if file is configured in ROI_MASKS
            for roi_name, roi_path in config.ROI_MASKS.items():
                if str(mask_file) == roi_path:
                    file_info['configured'] = True
                    file_info['roi_name'] = roi_name
                    break
            
            inventory_data.append(file_info)
            
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
    
    inventory_df = pd.DataFrame(inventory_data)
    inventory_df = inventory_df.sort_values('filename')
    
    print(f"Found {len(inventory_df)} mask files")
    print(f"Configured ROIs: {inventory_df['configured'].sum()}")
    print(f"Unconfigured files: {(~inventory_df['configured']).sum()}")
    
    return inventory_df


def main():
    """Main function for mask validation"""
    
    # Create argument parser
    parser = create_analysis_parser(
        script_name='validate_roi_masks',
        analysis_type='behavioral',  # closest match
        require_data=False
    )
    
    # Add script-specific arguments
    parser.parser.add_argument('--inventory', action='store_true',
                              help='Create inventory of all mask files')
    parser.parser.add_argument('--check-connectivity', action='store_true',
                              help='Check OAK storage connectivity only')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup environment - skip directory creation if just checking connectivity
    if args.check_connectivity:
        # Simple setup for connectivity check
        logger = setup_script_logging(
            script_name='validate_roi_masks',
            log_level='INFO',
            log_file=None  # No log file for connectivity check
        )
        config = OAKConfig()
        env = {'logger': logger, 'config': config}
    else:
        # Full setup for validation
        env = setup_pipeline_environment(
            script_name='validate_roi_masks',
            args=args,
            required_modules=['numpy', 'pandas', 'nibabel', 'nilearn', 'matplotlib']
        )
    
    logger = env['logger']
    config = env['config']
    
    try:
        if args.check_connectivity:
            # Check connectivity only
            logger.logger.info("Checking OAK connectivity...")
            oak_accessible = check_oak_connectivity()
            
            if oak_accessible:
                logger.logger.info("✓ OAK storage is accessible")
                logger.log_pipeline_end('validate_roi_masks', success=True)
            else:
                logger.logger.error("✗ OAK storage is not accessible")
                logger.log_pipeline_end('validate_roi_masks', success=False)
            
            return
        
        if args.inventory:
            # Create mask inventory
            logger.logger.info("Creating mask inventory...")
            inventory_df = create_mask_inventory(config)
            
            if not inventory_df.empty:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                inventory_file = output_dir / 'mask_inventory.csv'
                inventory_df.to_csv(inventory_file, index=False)
                
                logger.logger.info(f"Mask inventory saved: {inventory_file}")
                
                # Print summary
                print("\nMask Inventory Summary:")
                print(inventory_df[['filename', 'size_mb', 'configured', 'roi_name']].to_string(index=False))
            
            logger.log_pipeline_end('validate_roi_masks', success=True)
            return
        
        # Full validation
        logger.logger.info("Starting ROI mask validation...")
        
        # Check OAK connectivity first
        if not check_oak_connectivity():
            logger.logger.error("Cannot access OAK storage - validation aborted")
            logger.log_pipeline_end('validate_roi_masks', success=False)
            return
        
        # Create validator
        validator = MaskValidator(config)
        
        # Validate all masks
        results_df = validator.validate_all_masks()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations and reports
        validator.create_mask_visualizations(results_df, output_dir)
        validator.create_detailed_report(results_df, output_dir)
        
        # Log results
        analysis_results = {
            'success': validator.core_masks_valid,
            'core_masks_valid': validator.core_masks_valid,
            'optional_masks_available': len(validator.optional_masks_available),
            'output_dir': str(output_dir)
        }
        
        from logger_utils import log_analysis_results
        log_analysis_results(logger, analysis_results, 'mask_validation')
        
        if validator.core_masks_valid:
            logger.logger.info("✓ All core ROI masks are valid - pipeline ready!")
            available_rois = validator.get_available_rois()
            logger.logger.info(f"Available ROIs: {', '.join(available_rois)}")
        else:
            logger.logger.error("✗ Some core ROI masks are invalid - pipeline not ready")
        
        logger.log_pipeline_end('validate_roi_masks', success=validator.core_masks_valid)
        
    except Exception as e:
        logger.log_error_with_traceback(e, 'mask validation')
        logger.log_pipeline_end('validate_roi_masks', success=False)
        raise


if __name__ == "__main__":
    main() 