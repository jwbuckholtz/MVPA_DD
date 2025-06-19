#!/usr/bin/env python3
"""
Analyze and Visualize Results from Delay Discounting MVPA Pipeline

This script processes the results from the main analysis pipeline and creates
comprehensive visualizations and statistical summaries.

Author: Cognitive Neuroscience Lab, Stanford University
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsAnalyzer:
    """Class for analyzing and visualizing MVPA results"""
    
    def __init__(self, results_file):
        """
        Initialize with results file
        
        Parameters:
        -----------
        results_file : str
            Path to pickled results file
        """
        self.results_file = results_file
        self.results = self.load_results()
        self.output_dir = Path("./analysis_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_results(self):
        """Load results from pickle file"""
        with open(self.results_file, 'rb') as f:
            results = pickle.load(f)
        return results
    
    def extract_behavioral_summary(self):
        """Extract behavioral statistics across subjects"""
        behavioral_stats = []
        
        for subject_id, subject_data in self.results.items():
            if 'behavioral' in subject_data and subject_data['behavioral']['success']:
                behav_data = subject_data['behavioral']
                behavioral_stats.append({
                    'subject_id': subject_id,
                    'k': behav_data['k'],
                    'pseudo_r2': behav_data['pseudo_r2'],
                    'n_trials': behav_data['n_trials'],
                    'choice_rate': behav_data['choice_rate']
                })
        
        return pd.DataFrame(behavioral_stats)
    
    def extract_mvpa_summary(self):
        """Extract MVPA decoding results across subjects and ROIs"""
        mvpa_results = []
        
        for subject_id, subject_data in self.results.items():
            if 'mvpa' in subject_data:
                for roi_name, roi_results in subject_data['mvpa'].items():
                    if 'error' not in roi_results:
                        # Choice decoding
                        if ('choice_decoding' in roi_results and 
                            roi_results['choice_decoding']['success']):
                            choice_data = roi_results['choice_decoding']
                            mvpa_results.append({
                                'subject_id': subject_id,
                                'roi': roi_name,
                                'analysis_type': 'choice_decoding',
                                'metric': 'accuracy',
                                'value': choice_data['mean_accuracy'],
                                'std': choice_data['std_accuracy'],
                                'p_value': choice_data['p_value'],
                                'chance_level': choice_data['chance_level']
                            })
                        
                        # Continuous variable decoding
                        if 'continuous_decoding' in roi_results:
                            for var_name, var_results in roi_results['continuous_decoding'].items():
                                if var_results['success']:
                                    mvpa_results.append({
                                        'subject_id': subject_id,
                                        'roi': roi_name,
                                        'analysis_type': f'{var_name}_decoding',
                                        'metric': 'r2',
                                        'value': var_results['mean_r2'],
                                        'std': var_results['std_r2'],
                                        'p_value': var_results['p_value'],
                                        'chance_level': 0
                                    })
        
        return pd.DataFrame(mvpa_results)
    
    def extract_geometry_summary(self):
        """Extract neural geometry analysis results"""
        geometry_results = []
        
        for subject_id, subject_data in self.results.items():
            if 'geometry' in subject_data:
                for roi_name, roi_results in subject_data['geometry'].items():
                    if 'error' not in roi_results and 'correlations' in roi_results:
                        correlations = roi_results['correlations']
                        
                        for var_name, var_corr in correlations.items():
                            geometry_results.append({
                                'subject_id': subject_id,
                                'roi': roi_name,
                                'variable': var_name,
                                'max_correlation': var_corr['max_correlation'],
                                'best_dimension': var_corr['best_dimension'],
                                'min_p_value': np.min(var_corr['p_values'])
                            })
        
        return pd.DataFrame(geometry_results)
    
    def plot_behavioral_distributions(self, behavioral_df):
        """Plot distributions of behavioral parameters"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Discount rate (k) distribution
        axes[0, 0].hist(behavioral_df['k'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Discount Rate (k)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Discount Rates')
        axes[0, 0].axvline(behavioral_df['k'].median(), color='red', linestyle='--', 
                          label=f'Median = {behavioral_df["k"].median():.3f}')
        axes[0, 0].legend()
        
        # Model fit distribution
        axes[0, 1].hist(behavioral_df['pseudo_r2'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Pseudo R²')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Distribution of Model Fits')
        axes[0, 1].axvline(behavioral_df['pseudo_r2'].median(), color='red', linestyle='--',
                          label=f'Median = {behavioral_df["pseudo_r2"].median():.3f}')
        axes[0, 1].legend()
        
        # Choice rate distribution
        axes[1, 0].hist(behavioral_df['choice_rate'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Larger-Later Choice Rate')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Choice Rates')
        axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Chance Level')
        axes[1, 0].legend()
        
        # Correlation between k and choice rate
        axes[1, 1].scatter(behavioral_df['k'], behavioral_df['choice_rate'], alpha=0.7)
        r, p = stats.pearsonr(behavioral_df['k'], behavioral_df['choice_rate'])
        axes[1, 1].set_xlabel('Discount Rate (k)')
        axes[1, 1].set_ylabel('Larger-Later Choice Rate')
        axes[1, 1].set_title(f'k vs Choice Rate (r={r:.3f}, p={p:.3f})')
        
        # Add trend line
        z = np.polyfit(behavioral_df['k'], behavioral_df['choice_rate'], 1)
        p_trend = np.poly1d(z)
        axes[1, 1].plot(behavioral_df['k'], p_trend(behavioral_df['k']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'behavioral_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Behavioral distributions saved: {self.output_dir / 'behavioral_distributions.png'}")
    
    def plot_mvpa_results(self, mvpa_df):
        """Plot MVPA decoding results"""
        # Group-level statistics
        group_stats = []
        
        for analysis_type in mvpa_df['analysis_type'].unique():
            for roi in mvpa_df['roi'].unique():
                subset = mvpa_df[(mvpa_df['analysis_type'] == analysis_type) & 
                               (mvpa_df['roi'] == roi)]
                
                if len(subset) > 0:
                    # Test against chance
                    if 'choice' in analysis_type:
                        chance_level = subset['chance_level'].iloc[0]
                    else:
                        chance_level = 0
                    
                    t_stat, p_val = stats.ttest_1samp(subset['value'], chance_level)
                    
                    group_stats.append({
                        'analysis_type': analysis_type,
                        'roi': roi,
                        'mean_accuracy': subset['value'].mean(),
                        'std_accuracy': subset['value'].std(),
                        'n_subjects': len(subset),
                        't_stat': t_stat,
                        'p_value': p_val,
                        'chance_level': chance_level
                    })
        
        group_stats_df = pd.DataFrame(group_stats)
        
        if len(group_stats_df) > 0:
            # Multiple comparisons correction
            _, group_stats_df['p_corrected'], _, _ = multipletests(
                group_stats_df['p_value'], method='fdr_bh'
            )
            
            # Save group statistics
            group_stats_df.to_csv(self.output_dir / 'group_mvpa_statistics.csv', index=False)
            print(f"MVPA group statistics saved: {self.output_dir / 'group_mvpa_statistics.csv'}")
        
        return group_stats_df
    
    def plot_group_embeddings(self, geometry_df):
        """Plot group-level embedding analyses"""
        if len(geometry_df) == 0:
            print("No geometry results to plot")
            return
        
        # Plot maximum correlations by ROI and variable
        pivot_corr = geometry_df.pivot_table(
            index='roi', columns='variable', values='max_correlation', aggfunc='mean'
        )
        
        if len(pivot_corr) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_corr, annot=True, fmt='.3f', cmap='RdBu_r', 
                       center=0, ax=ax, cbar_kws={'label': 'Max Correlation'})
            
            ax.set_title('Neural Geometry Correlations with Behavioral Variables')
            ax.set_xlabel('Behavioral Variable')
            ax.set_ylabel('ROI')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'group_geometry_correlations.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Group geometry correlations saved: {self.output_dir / 'group_geometry_correlations.png'}")
        
        # Plot distribution of correlations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        variables = geometry_df['variable'].unique()
        for i, var in enumerate(variables[:4]):  # Plot first 4 variables
            if i < len(axes):
                var_data = geometry_df[geometry_df['variable'] == var]
                axes[i].hist(var_data['max_correlation'], bins=15, alpha=0.7, 
                           edgecolor='black')
                axes[i].set_xlabel('Max Correlation')
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'{var} Correlations')
                axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)
                
                # Add statistics
                mean_corr = var_data['max_correlation'].mean()
                axes[i].axvline(mean_corr, color='blue', linestyle='-', alpha=0.7, 
                              label=f'Mean: {mean_corr:.3f}')
                axes[i].legend()
        
        # Remove unused subplots
        for i in range(len(variables), len(axes)):
            if i < len(axes):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'group_geometry_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Group geometry distributions saved: {self.output_dir / 'group_geometry_distributions.png'}")
    
    def create_summary_report(self, behavioral_df, mvpa_stats_df, geometry_df):
        """Create comprehensive summary report"""
        report_lines = []
        report_lines.append("Delay Discounting MVPA Analysis - Summary Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Behavioral summary
        report_lines.append("BEHAVIORAL ANALYSIS SUMMARY:")
        report_lines.append(f"Number of subjects: {len(behavioral_df)}")
        if len(behavioral_df) > 0:
            report_lines.append(f"Mean discount rate (k): {behavioral_df['k'].mean():.4f} ± {behavioral_df['k'].std():.4f}")
            report_lines.append(f"Mean model fit (pseudo R²): {behavioral_df['pseudo_r2'].mean():.3f} ± {behavioral_df['pseudo_r2'].std():.3f}")
            report_lines.append(f"Mean choice rate: {behavioral_df['choice_rate'].mean():.3f} ± {behavioral_df['choice_rate'].std():.3f}")
        report_lines.append("")
        
        # MVPA summary
        report_lines.append("MVPA DECODING SUMMARY:")
        if len(mvpa_stats_df) > 0:
            choice_results = mvpa_stats_df[mvpa_stats_df['analysis_type'] == 'choice_decoding']
            if len(choice_results) > 0:
                report_lines.append("Choice Decoding Results:")
                for _, row in choice_results.iterrows():
                    significance = ""
                    if row['p_corrected'] < 0.001:
                        significance = "***"
                    elif row['p_corrected'] < 0.01:
                        significance = "**"
                    elif row['p_corrected'] < 0.05:
                        significance = "*"
                    
                    report_lines.append(f"  {row['roi']}: {row['mean_accuracy']:.3f} ± {row['std_accuracy']:.3f} {significance}")
        else:
            report_lines.append("No MVPA results available")
        
        report_lines.append("")
        
        # Geometry summary
        if len(geometry_df) > 0:
            report_lines.append("NEURAL GEOMETRY SUMMARY:")
            report_lines.append(f"Number of geometry analyses: {len(geometry_df)}")
            
            # Strongest correlations
            strongest_corr = geometry_df.loc[geometry_df['max_correlation'].idxmax()]
            report_lines.append(f"Strongest correlation: {strongest_corr['variable']} in {strongest_corr['roi']} (r = {strongest_corr['max_correlation']:.3f})")
        else:
            report_lines.append("NEURAL GEOMETRY SUMMARY:")
            report_lines.append("No geometry results available")
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"Summary report saved: {self.output_dir / 'summary_report.txt'}")
        print("\nSUMMARY:")
        print(report_text)
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting results analysis...")
        
        # Extract data summaries
        behavioral_df = self.extract_behavioral_summary()
        mvpa_df = self.extract_mvpa_summary()
        geometry_df = self.extract_geometry_summary()
        
        print(f"Extracted data for {len(behavioral_df)} subjects")
        print(f"MVPA results: {len(mvpa_df)} analyses")
        print(f"Geometry results: {len(geometry_df)} analyses")
        
        # Create visualizations
        mvpa_stats_df = pd.DataFrame()
        
        if len(behavioral_df) > 0:
            self.plot_behavioral_distributions(behavioral_df)
        
        if len(mvpa_df) > 0:
            mvpa_stats_df = self.plot_mvpa_results(mvpa_df)
        
        if len(geometry_df) > 0:
            self.plot_group_embeddings(geometry_df)
        
        # Create summary report
        self.create_summary_report(behavioral_df, mvpa_stats_df, geometry_df)
        
        print("Analysis complete!")

def main():
    """Main function"""
    # Look for results file
    results_file = "./delay_discounting_results/all_results.pkl"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run the main analysis pipeline first.")
        return
    
    # Run analysis
    analyzer = ResultsAnalyzer(results_file)
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 