#!/usr/bin/env python3
"""
Demo Script: ROI Mask Validation
================================

This script demonstrates the new mask validation functionality
for the delay discounting MVPA pipeline.

Usage:
    python demo_mask_validation.py
    
Author: Cognitive Neuroscience Lab, Stanford University
"""

# Import logger utilities for standardized setup
from logger_utils import setup_script_logging, PipelineLogger

import sys
from pathlib import Path
import pandas as pd

# Core modules
from oak_storage_config import OAKConfig
from validate_roi_masks import MaskValidator, check_oak_connectivity, create_mask_inventory
from data_utils import check_mask_exists, load_mask, DataError


def demo_oak_connectivity():
    """Demo OAK connectivity check"""
    print("=" * 60)
    print("DEMO 1: OAK CONNECTIVITY CHECK")
    print("=" * 60)
    
    print("Checking if OAK storage is accessible...")
    
    config = OAKConfig()
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Masks directory: {config.MASKS_DIR}")
    
    # Check connectivity
    oak_accessible = check_oak_connectivity()
    
    if oak_accessible:
        print("✓ OAK storage is accessible")
    else:
        print("✗ OAK storage is NOT accessible")
        print("Note: This is expected if running outside Stanford network")
    
    return oak_accessible


def demo_mask_inventory():
    """Demo mask inventory creation"""
    print("\n" + "=" * 60)
    print("DEMO 2: MASK INVENTORY")
    print("=" * 60)
    
    config = OAKConfig()
    
    print("Creating inventory of all mask files...")
    
    try:
        inventory_df = create_mask_inventory(config)
        
        if inventory_df.empty:
            print("No mask files found or directory not accessible")
            return False
        
        print(f"Found {len(inventory_df)} mask files:")
        print(inventory_df[['filename', 'size_mb', 'configured', 'roi_name']].to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"Error creating inventory: {e}")
        return False


def demo_mask_validation():
    """Demo comprehensive mask validation"""
    print("\n" + "=" * 60)
    print("DEMO 3: COMPREHENSIVE MASK VALIDATION")
    print("=" * 60)
    
    config = OAKConfig()
    
    print("Running comprehensive mask validation...")
    
    try:
        # Create validator
        validator = MaskValidator(config)
        
        # Validate all masks
        results_df = validator.validate_all_masks()
        
        # Show results
        print(f"\nValidation Results Summary:")
        print(f"Core masks configured: {len(config.CORE_ROI_MASKS)}")
        print(f"Optional masks configured: {len(config.OPTIONAL_ROI_MASKS)}")
        print(f"Total masks to validate: {len(results_df)}")
        
        if not results_df.empty:
            valid_count = results_df['valid'].sum()
            print(f"Valid masks: {valid_count}/{len(results_df)}")
            
            print("\nDetailed Results:")
            for _, row in results_df.iterrows():
                status = "✓" if row['valid'] else "✗"
                mask_type = row['mask_type'].upper()
                print(f"  {status} {mask_type}: {row['roi_name']}")
                if row['valid']:
                    print(f"    └─ {row['n_voxels']} voxels")
                else:
                    print(f"    └─ {row['error']}")
        
        print(f"\nPipeline ready: {'YES' if validator.core_masks_valid else 'NO'}")
        
        if validator.core_masks_valid:
            available_rois = validator.get_available_rois()
            print(f"Available ROIs: {', '.join(available_rois)}")
        
        return validator.core_masks_valid
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return False


def demo_individual_mask_check():
    """Demo individual mask checking"""
    print("\n" + "=" * 60)
    print("DEMO 4: INDIVIDUAL MASK CHECKING")
    print("=" * 60)
    
    config = OAKConfig()
    
    print("Checking individual masks using data_utils...")
    
    # Check each configured mask
    for roi_name, mask_path in config.ROI_MASKS.items():
        print(f"\nChecking {roi_name}:")
        print(f"  Path: {mask_path}")
        
        # Check existence
        exists = check_mask_exists(mask_path)
        print(f"  Exists: {'✓' if exists else '✗'}")
        
        if exists:
            try:
                # Load and validate
                mask_img = load_mask(mask_path, validate=True)
                mask_data = mask_img.get_fdata()
                
                n_voxels = (mask_data > 0).sum()
                dimensions = mask_data.shape
                resolution = mask_img.header.get_zooms()[:3]
                
                print(f"  ✓ Valid: {n_voxels} voxels")
                print(f"  Dimensions: {dimensions}")
                print(f"  Resolution: {resolution}")
                
            except DataError as e:
                print(f"  ✗ Validation failed: {e}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print(f"  ✗ Not found")


def demo_mask_visualization():
    """Demo mask visualization creation"""
    print("\n" + "=" * 60)
    print("DEMO 5: MASK VISUALIZATION")
    print("=" * 60)
    
    config = OAKConfig()
    
    print("Creating mask visualizations...")
    
    try:
        # Create validator
        validator = MaskValidator(config)
        
        # Validate masks
        results_df = validator.validate_all_masks()
        
        # Create output directory
        output_dir = Path("./demo_mask_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        validator.create_mask_visualizations(results_df, output_dir)
        validator.create_detailed_report(results_df, output_dir)
        
        print(f"\nOutputs created in: {output_dir}")
        print("Files generated:")
        for file in output_dir.glob("*"):
            print(f"  - {file.name}")
        
        return True
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return False


def main():
    """Main demo function"""
    print("ROI Mask Validation Demo")
    print("=" * 60)
    print("This demo shows the new mask validation functionality")
    print("for the delay discounting MVPA pipeline.")
    print()
    
    # Setup logging
    logger = setup_script_logging(
        script_name='demo_mask_validation',
        log_level='INFO',
        log_file='demo_mask_validation.log'
    )
    
    logger.logger.info("Starting mask validation demo")
    
    # Run demos
    try:
        # Demo 1: OAK connectivity
        oak_accessible = demo_oak_connectivity()
        
        if not oak_accessible:
            print("\n" + "⚠️  WARNING: OAK storage not accessible")
            print("Remaining demos will show expected behavior with mock data")
        
        # Demo 2: Mask inventory
        inventory_success = demo_mask_inventory()
        
        # Demo 3: Comprehensive validation
        validation_success = demo_mask_validation()
        
        # Demo 4: Individual mask checking
        demo_individual_mask_check()
        
        # Demo 5: Visualization (if masks are available)
        if oak_accessible and validation_success:
            demo_mask_visualization()
        else:
            print("\n" + "=" * 60)
            print("DEMO 5: MASK VISUALIZATION")
            print("=" * 60)
            print("Skipping visualization demo - masks not accessible")
        
        # Summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"OAK accessible: {'✓' if oak_accessible else '✗'}")
        print(f"Inventory created: {'✓' if inventory_success else '✗'}")
        print(f"Validation successful: {'✓' if validation_success else '✗'}")
        
        if oak_accessible and validation_success:
            print("\n✓ All demos completed successfully!")
            print("The pipeline is ready to use pre-existing masks on OAK.")
        else:
            print("\n⚠️  Some demos failed due to OAK accessibility")
            print("This is expected when running outside Stanford network.")
        
        logger.logger.info("Mask validation demo completed")
        
    except Exception as e:
        logger.logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 