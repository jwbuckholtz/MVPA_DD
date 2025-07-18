
import argparse
from behavioral_analysis import BehavioralAnalysis
from mvpa_analysis import MVPAAnalysis
from geometry_analysis import GeometryAnalysis
import config

def main():
    """
    Main entrypoint for the fMRI analysis pipeline.
    This script orchestrates the behavioral, MVPA, and geometry analyses.
    """
    parser = argparse.ArgumentParser(description="fMRI Analysis Pipeline")
    parser.add_argument('--subjects', nargs='+', help='A list of subject IDs to process.')
    parser.add_argument('--skip-behavioral', action='store_true', help='Skip the behavioral analysis.')
    parser.add_argument('--skip-mvpa', action='store_true', help='Skip the MVPA analysis.')
    parser.add_argument('--skip-geometry', action='store_true', help='Skip the geometry analysis.')
    args = parser.parse_args()

    # Use a default subject list if none are provided
    subjects = args.subjects if args.subjects else ['s001', 's002', 's003']

    # --- Behavioral Analysis ---
    if not args.skip_behavioral:
        print("--- Running Behavioral Analysis ---")
        behavioral_analysis = BehavioralAnalysis()
        behavioral_analysis.run_analysis(subjects)
        print("--- Behavioral Analysis Complete ---")

    # --- MVPA Analysis ---
    if not args.skip_mvpa:
        print("\n--- Running MVPA Analysis ---")
        mvpa_analysis = MVPAAnalysis()
        mvpa_analysis.run_analysis(subjects)
        print("--- MVPA Analysis Complete ---")

    # --- Geometry Analysis ---
    if not args.skip_geometry:
        print("\n--- Running Geometry Analysis ---")
        geometry_analysis = GeometryAnalysis()
        geometry_analysis.run_analysis(subjects)
        print("--- Geometry Analysis Complete ---")

if __name__ == '__main__':
    main() 