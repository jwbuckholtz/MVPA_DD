#!/bin/bash
#SBATCH --job-name=mvpa_analysis
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=mvpa_analysis_%j.out
#SBATCH --error=mvpa_analysis_%j.err

# Load necessary modules
module load python/3.9.0
module load py-numpy/1.23.0
module load py-scipy/1.9.0

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the MVPA analysis
python mvpa_delay_discounting.py 