#!/usr/bin/env python3
"""
Create ROI masks for MVPA analysis
This script creates anatomical ROI masks for:
- Striatum (caudate, putamen, nucleus accumbens)
- DLPFC (Brodmann areas 9, 46)  
- VMPFC (Brodmann areas 10, 11, 32)

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import datasets, image, plotting
from nilearn.regions import RegionExtractor
import matplotlib.pyplot as plt

def create_output_directory():
    """Create masks directory"""
    masks_dir = Path('./masks')
    masks_dir.mkdir(exist_ok=True)
    return masks_dir

def create_striatum_mask(masks_dir):
    """
    Create striatum mask using Harvard-Oxford subcortical atlas
    Includes: Caudate, Putamen, Nucleus Accumbens
    """
    print("Creating striatum mask...")
    
    # Load Harvard-Oxford subcortical atlas
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    atlas_img = ho_sub.maps
    labels = ho_sub.labels
    
    # Find indices for striatal regions
    striatal_regions = ['Left Caudate', 'Right Caudate', 
                       'Left Putamen', 'Right Putamen',
                       'Left Accumbens', 'Right Accumbens']
    
    indices = []
    for region in striatal_regions:
        if region in labels:
            indices.append(labels.index(region))
    
    print(f"Found striatal regions at indices: {indices}")
    
    # Create binary mask
    atlas_data = atlas_img.get_fdata()
    striatum_mask = np.zeros_like(atlas_data)
    
    for idx in indices:
        striatum_mask[atlas_data == idx] = 1
    
    # Save mask
    striatum_img = nib.Nifti1Image(striatum_mask, atlas_img.affine, atlas_img.header)
    mask_file = masks_dir / 'striatum_mask.nii.gz'
    nib.save(striatum_img, str(mask_file))
    
    print(f"Striatum mask saved to {mask_file}")
    return str(mask_file)

def create_dlpfc_mask(masks_dir):
    """
    Create DLPFC mask using Harvard-Oxford cortical atlas
    Includes: Middle and Superior Frontal Gyrus regions
    """
    print("Creating DLPFC mask...")
    
    # Load Harvard-Oxford cortical atlas
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = ho_cort.maps
    labels = ho_cort.labels
    
    # Find indices for DLPFC regions
    dlpfc_regions = ['Left Middle Frontal Gyrus', 'Right Middle Frontal Gyrus',
                    'Left Superior Frontal Gyrus', 'Right Superior Frontal Gyrus']
    
    indices = []
    for region in dlpfc_regions:
        if region in labels:
            indices.append(labels.index(region))
    
    print(f"Found DLPFC regions at indices: {indices}")
    
    # Create binary mask
    atlas_data = atlas_img.get_fdata()
    dlpfc_mask = np.zeros_like(atlas_data)
    
    for idx in indices:
        dlpfc_mask[atlas_data == idx] = 1
    
    # Save mask
    dlpfc_img = nib.Nifti1Image(dlpfc_mask, atlas_img.affine, atlas_img.header)
    mask_file = masks_dir / 'dlpfc_mask.nii.gz'
    nib.save(dlpfc_img, str(mask_file))
    
    print(f"DLPFC mask saved to {mask_file}")
    return str(mask_file)

def create_vmpfc_mask(masks_dir):
    """
    Create VMPFC mask using Harvard-Oxford cortical atlas
    Includes: Frontal Medial Cortex, Frontal Orbital Cortex, Subcallosal Cortex
    """
    print("Creating VMPFC mask...")
    
    # Load Harvard-Oxford cortical atlas
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = ho_cort.maps
    labels = ho_cort.labels
    
    # Find indices for VMPFC regions
    vmpfc_regions = ['Frontal Medial Cortex',
                    'Left Frontal Orbital Cortex', 'Right Frontal Orbital Cortex',
                    'Left Subcallosal Cortex', 'Right Subcallosal Cortex']
    
    indices = []
    for region in vmpfc_regions:
        if region in labels:
            indices.append(labels.index(region))
    
    print(f"Found VMPFC regions at indices: {indices}")
    
    # Create binary mask
    atlas_data = atlas_img.get_fdata()
    vmpfc_mask = np.zeros_like(atlas_data)
    
    for idx in indices:
        vmpfc_mask[atlas_data == idx] = 1
    
    # Save mask
    vmpfc_img = nib.Nifti1Image(vmpfc_mask, atlas_img.affine, atlas_img.header)
    mask_file = masks_dir / 'vmpfc_mask.nii.gz'
    nib.save(vmpfc_img, str(mask_file))
    
    print(f"VMPFC mask saved to {mask_file}")
    return str(mask_file)

def create_custom_dlpfc_mask(masks_dir):
    """
    Create more precise DLPFC mask using coordinates from literature
    Based on meta-analysis coordinates for BA 9/46
    """
    print("Creating custom DLPFC mask from coordinates...")
    
    # DLPFC coordinates from meta-analyses (MNI space)
    # Left DLPFC
    left_coords = [
        [-42, 36, 24],   # BA 9/46 border
        [-36, 48, 12],   # BA 10/46 border  
        [-48, 24, 36],   # BA 9
        [-42, 12, 48]    # BA 9
    ]
    
    # Right DLPFC
    right_coords = [
        [42, 36, 24],    # BA 9/46 border
        [36, 48, 12],    # BA 10/46 border
        [48, 24, 36],    # BA 9  
        [42, 12, 48]     # BA 9
    ]
    
    # Load template for affine transform
    template = datasets.load_mni152_template(resolution=2)
    
    # Create spherical masks around coordinates
    from nilearn.image import new_img_like
    
    mask_data = np.zeros(template.shape)
    
    # Function to create sphere around coordinate
    def create_sphere(center, radius=10):
        x, y, z = np.mgrid[0:mask_data.shape[0], 
                          0:mask_data.shape[1], 
                          0:mask_data.shape[2]]
        
        # Convert voxel coordinates to mm
        coords_mm = image.coord_transform(x, y, z, template.affine)
        
        # Calculate distance from center
        distances = np.sqrt((coords_mm[0] - center[0])**2 + 
                           (coords_mm[1] - center[1])**2 + 
                           (coords_mm[2] - center[2])**2)
        
        return distances <= radius
    
    # Add spheres for each coordinate
    all_coords = left_coords + right_coords
    for coord in all_coords:
        sphere = create_sphere(coord, radius=8)  # 8mm radius
        mask_data[sphere] = 1
    
    # Save mask
    dlpfc_custom_img = new_img_like(template, mask_data)
    mask_file = masks_dir / 'dlpfc_custom_mask.nii.gz'
    nib.save(dlpfc_custom_img, str(mask_file))
    
    print(f"Custom DLPFC mask saved to {mask_file}")
    return str(mask_file)

def visualize_masks(mask_files):
    """Create visualization of all ROI masks"""
    print("Creating mask visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    mask_names = ['Striatum', 'DLPFC', 'VMPFC']
    
    for i, (mask_file, name) in enumerate(zip(mask_files, mask_names)):
        plotting.plot_roi(mask_file, 
                         title=f'{name} Mask',
                         axes=axes[i],
                         cmap='Reds')
    
    # Remove empty subplot
    if len(mask_files) < 4:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('./masks/roi_masks_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Mask visualization saved to ./masks/roi_masks_visualization.png")

def main():
    """Main function to create all ROI masks"""
    print("Creating ROI Masks for Delay Discounting MVPA Analysis")
    print("=" * 60)
    
    # Create output directory
    masks_dir = create_output_directory()
    
    # Create masks
    mask_files = []
    
    # Striatum
    striatum_file = create_striatum_mask(masks_dir)
    mask_files.append(striatum_file)
    
    # DLPFC
    dlpfc_file = create_dlpfc_mask(masks_dir)
    mask_files.append(dlpfc_file)
    
    # Custom DLPFC (optional, more precise)
    # dlpfc_custom_file = create_custom_dlpfc_mask(masks_dir)
    
    # VMPFC
    vmpfc_file = create_vmpfc_mask(masks_dir)
    mask_files.append(vmpfc_file)
    
    # Create visualizations
    visualize_masks(mask_files)
    
    print("\nMask creation complete!")
    print(f"Created {len(mask_files)} ROI masks:")
    for mask_file in mask_files:
        print(f"  - {mask_file}")

if __name__ == "__main__":
    main() 