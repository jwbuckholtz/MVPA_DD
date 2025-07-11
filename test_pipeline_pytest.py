#!/usr/bin/env python3
"""
Comprehensive Unit Tests for MVPA Pipeline
==========================================

This module provides comprehensive unit tests for all components of the MVPA pipeline
using pytest. Tests cover:

1. Analysis Base Classes (BaseAnalysis, AnalysisFactory)
2. Behavioral Analysis (hyperbolic discounting, validation)
3. MVPA Analysis (pattern extraction, decoding, cross-validation)
4. Geometry Analysis (RDM, dimensionality reduction, comparisons)
5. Mask Creation and Validation
6. Data Loading and Utilities
7. Error Handling and Edge Cases

Author: Cognitive Neuroscience Lab, Stanford University
"""

import pytest
import numpy as np
import pandas as pd
import nibabel as nib
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings
from typing import Dict, List, Any
import pickle

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Test data constants
TEST_SUBJECTS = ['test_subject_001', 'test_subject_002']
TEST_ROIS = ['test_roi_1', 'test_roi_2']
N_TRIALS = 50
N_VOXELS = 100
N_TIMEPOINTS = 200


# ============================================================================
# Fixtures for Common Test Data
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a mock configuration object"""
    config = Mock()
    config.TR = 2.0
    config.HEMI_LAG = 4.0
    config.CV_FOLDS = 5
    config.N_PERMUTATIONS = 100
    config.OUTPUT_DIR = "/tmp/test_output"
    config.RESULTS_DIR = "/tmp/test_results"
    config.ROI_MASKS = {
        'test_roi_1': '/path/to/roi1.nii.gz',
        'test_roi_2': '/path/to/roi2.nii.gz'
    }
    return config


@pytest.fixture
def synthetic_behavioral_data():
    """Create synthetic behavioral data for testing"""
    np.random.seed(42)
    
    data = {
        'trial_number': np.arange(N_TRIALS),
        'choice': np.random.randint(0, 2, N_TRIALS),
        'large_amount': np.random.uniform(25, 100, N_TRIALS),
        'delay_days': np.random.uniform(1, 365, N_TRIALS),
        'sv_diff': np.random.normal(0, 10, N_TRIALS),
        'sv_sum': np.random.uniform(20, 120, N_TRIALS),
        'sv_chosen': np.random.uniform(10, 80, N_TRIALS),
        'sv_unchosen': np.random.uniform(5, 70, N_TRIALS),
        'onset': np.cumsum(np.random.uniform(8, 12, N_TRIALS))
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_fmri_data():
    """Create synthetic fMRI data for testing"""
    np.random.seed(42)
    
    # Create 4D data (x, y, z, t)
    data = np.random.randn(20, 20, 20, N_TIMEPOINTS)
    
    # Create affine matrix
    affine = np.eye(4)
    affine[:3, :3] *= 2.0  # 2mm voxels
    
    # Create nibabel image
    img = nib.Nifti1Image(data, affine)
    
    return img


@pytest.fixture
def synthetic_neural_patterns():
    """Create synthetic neural pattern data"""
    np.random.seed(42)
    return np.random.randn(N_TRIALS, N_VOXELS)


@pytest.fixture
def synthetic_confounds():
    """Create synthetic confound data"""
    np.random.seed(42)
    n_confounds = 6
    return np.random.randn(N_TIMEPOINTS, n_confounds)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# ============================================================================
# Test Analysis Base Classes
# ============================================================================

class TestAnalysisBase:
    """Test the base analysis classes and factory"""
    
    def test_analysis_factory_creation(self, mock_config):
        """Test AnalysisFactory can create all analysis types"""
        from analysis_base import AnalysisFactory
        
        # Test available types
        available_types = AnalysisFactory.list_available()
        assert 'behavioral' in available_types
        assert 'mvpa' in available_types
        assert 'geometry' in available_types
        
        # Test creation
        behavioral_analysis = AnalysisFactory.create('behavioral', config=mock_config)
        assert behavioral_analysis.name == 'BehavioralAnalysis'
        assert behavioral_analysis.config == mock_config
        
        mvpa_analysis = AnalysisFactory.create('mvpa', config=mock_config)
        assert mvpa_analysis.name == 'MVPAAnalysis'
        
        geometry_analysis = AnalysisFactory.create('geometry', config=mock_config)
        assert geometry_analysis.name == 'GeometryAnalysis'
    
    def test_analysis_factory_unknown_type(self):
        """Test AnalysisFactory raises error for unknown type"""
        from analysis_base import AnalysisFactory
        
        with pytest.raises(ValueError, match="Unknown analysis type"):
            AnalysisFactory.create('unknown_type')
    
    def test_convenience_function(self, mock_config):
        """Test convenience function for creating analysis"""
        from analysis_base import create_analysis
        
        analysis = create_analysis('behavioral', config=mock_config)
        assert analysis.name == 'BehavioralAnalysis'
    
    def test_base_analysis_initialization(self, mock_config):
        """Test BaseAnalysis initialization"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Check initialization
        assert analysis.name == 'BehavioralAnalysis'
        assert analysis.config == mock_config
        assert analysis.results == {}
        assert analysis.processing_stats['subjects_processed'] == 0
        assert analysis.processing_stats['subjects_failed'] == 0
        assert len(analysis._data_cache) == 0
    
    def test_base_analysis_result_handling(self, mock_config, temp_output_dir):
        """Test result saving and loading"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Add some fake results
        analysis.results = {
            'subject_001': {'success': True, 'k': 0.025},
            'subject_002': {'success': True, 'k': 0.018}
        }
        
        # Test saving
        results_path = Path(temp_output_dir) / 'test_results.pkl'
        saved_path = analysis.save_results(str(results_path))
        assert Path(saved_path).exists()
        
        # Test loading
        new_analysis = BehavioralAnalysis(config=mock_config)
        loaded_data = new_analysis.load_results(saved_path)
        
        assert 'results' in loaded_data
        assert len(new_analysis.results) == 2
        assert new_analysis.results['subject_001']['k'] == 0.025
    
    def test_base_analysis_cache_management(self, mock_config):
        """Test data cache management"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Add some data to cache
        analysis._data_cache['test_key'] = 'test_data'
        
        # Check cache info
        cache_info = analysis.get_cache_info()
        assert cache_info['cache_size'] == 1
        assert 'test_key' in cache_info['cached_keys']
        
        # Clear cache
        analysis.clear_cache()
        assert len(analysis._data_cache) == 0


# ============================================================================
# Test Behavioral Analysis
# ============================================================================

class TestBehavioralAnalysis:
    """Test behavioral analysis components"""
    
    def test_hyperbolic_discount_function(self, mock_config):
        """Test hyperbolic discount function"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Test with single values
        delay = np.array([1, 7, 30, 90])
        k = 0.01
        values = analysis.hyperbolic_discount_function(delay, k)
        
        assert len(values) == 4
        assert np.all(values > 0)
        assert np.all(values <= 1)
        
        # Test that values decrease with delay
        assert values[0] > values[1] > values[2] > values[3]
        
        # Test edge cases
        assert analysis.hyperbolic_discount_function(np.array([0]), k)[0] == 1.0
        
        # Test with k=0 (no discounting)
        no_discount = analysis.hyperbolic_discount_function(delay, 0)
        assert np.allclose(no_discount, 1.0)
    
    def test_subjective_value_calculation(self, mock_config):
        """Test subjective value calculation"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        amounts = np.array([20, 30, 40, 50])
        delays = np.array([1, 7, 30, 90])
        k = 0.01
        
        sv = analysis.subjective_value(amounts, delays, k)
        
        assert len(sv) == 4
        assert np.all(sv > 0)
        assert np.all(sv <= amounts)  # SV should be less than or equal to amount
        
        # Test that immediate rewards have full value
        immediate_sv = analysis.subjective_value(amounts, np.zeros(4), k)
        assert np.allclose(immediate_sv, amounts)
    
    def test_discount_rate_fitting(self, mock_config):
        """Test discount rate fitting"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Create test data with known discount rate
        k_true = 0.02
        large_amounts = np.array([25, 30, 40, 50, 60])
        delays = np.array([1, 7, 30, 90, 180])
        
        # Generate choices based on subjective values
        sv_large = analysis.subjective_value(large_amounts, delays, k_true)
        sv_small = 20.0  # Immediate smaller reward
        
        # Probabilistic choices (add some noise)
        np.random.seed(42)
        choice_probs = 1 / (1 + np.exp(-(sv_large - sv_small)))
        choices = np.random.binomial(1, choice_probs)
        
        # Fit discount rate
        fit_result = analysis.fit_discount_rate(choices, large_amounts, delays)
        
        assert fit_result['success'] == True
        assert 'k' in fit_result
        assert 'pseudo_r2' in fit_result
        assert 'choice_prob' in fit_result
        
        # Check that fitted k is reasonable
        assert fit_result['k'] > 0
        assert fit_result['k'] < 1
        assert fit_result['pseudo_r2'] >= 0
    
    def test_discount_rate_fitting_edge_cases(self, mock_config):
        """Test discount rate fitting with edge cases"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Test with no choice variability (all same choice)
        choices = np.ones(5)
        amounts = np.array([25, 30, 40, 50, 60])
        delays = np.array([1, 7, 30, 90, 180])
        
        fit_result = analysis.fit_discount_rate(choices, amounts, delays)
        # Should still work but might have poor fit
        assert fit_result['success'] == True
        
        # Test with very few trials
        fit_result = analysis.fit_discount_rate(choices[:2], amounts[:2], delays[:2])
        assert fit_result['success'] == True
    
    def test_behavioral_data_validation(self, mock_config, synthetic_behavioral_data):
        """Test behavioral data validation"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Test with valid data
        validation_result = analysis.validate_behavioral_data(synthetic_behavioral_data)
        assert validation_result['valid'] == True
        assert 'n_trials' in validation_result['stats']
        
        # Test with insufficient trials
        small_data = synthetic_behavioral_data.iloc[:5]
        validation_result = analysis.validate_behavioral_data(small_data)
        assert validation_result['valid'] == False
        assert 'Insufficient trials' in validation_result['errors'][0]
        
        # Test with missing columns
        incomplete_data = synthetic_behavioral_data.drop('choice', axis=1)
        validation_result = analysis.validate_behavioral_data(incomplete_data)
        assert validation_result['valid'] == False
        assert 'Missing required columns' in validation_result['errors'][0]
        
        # Test with no choice variability
        no_var_data = synthetic_behavioral_data.copy()
        no_var_data['choice'] = 1
        validation_result = analysis.validate_behavioral_data(no_var_data)
        assert validation_result['valid'] == False
        assert 'No choice variability' in validation_result['errors'][0]
    
    @patch('behavioral_analysis.BehavioralAnalysis.load_behavioral_data')
    def test_process_subject_success(self, mock_load_data, mock_config, synthetic_behavioral_data):
        """Test successful subject processing"""
        from behavioral_analysis import BehavioralAnalysis
        
        # Mock data loading
        mock_load_data.return_value = synthetic_behavioral_data
        
        analysis = BehavioralAnalysis(config=mock_config)
        result = analysis.process_subject('test_subject_001')
        
        assert result['success'] == True
        assert 'k' in result
        assert 'pseudo_r2' in result
        assert result['subject_id'] == 'test_subject_001'
        assert result['n_trials'] == N_TRIALS
        assert result['processing_time'] > 0
    
    @patch('behavioral_analysis.BehavioralAnalysis.load_behavioral_data')
    def test_process_subject_failure(self, mock_load_data, mock_config):
        """Test subject processing failure"""
        from behavioral_analysis import BehavioralAnalysis
        from analysis_base import AnalysisError
        
        # Mock data loading to raise error
        mock_load_data.side_effect = AnalysisError("Data loading failed")
        
        analysis = BehavioralAnalysis(config=mock_config)
        result = analysis.process_subject('test_subject_001')
        
        assert result['success'] == False
        assert 'error' in result
        assert 'Data loading failed' in result['error']
    
    def test_create_summary_dataframe(self, mock_config):
        """Test behavioral summary dataframe creation"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Add some fake results
        analysis.results = {
            'subject_001': {
                'success': True,
                'k': 0.025,
                'pseudo_r2': 0.45,
                'n_trials': 100,
                'n_valid_trials': 95,
                'choice_rate': 0.6,
                'processing_time': 2.3
            },
            'subject_002': {
                'success': True,
                'k': 0.018,
                'pseudo_r2': 0.52,
                'n_trials': 98,
                'n_valid_trials': 92,
                'choice_rate': 0.55,
                'processing_time': 2.1
            }
        }
        
        df = analysis.create_behavioral_summary_dataframe()
        
        assert len(df) == 2
        assert 'subject_id' in df.columns
        assert 'k' in df.columns
        assert 'pseudo_r2' in df.columns
        assert df['k'].iloc[0] == 0.025
        assert df['k'].iloc[1] == 0.018


# ============================================================================
# Test MVPA Analysis
# ============================================================================

class TestMVPAAnalysis:
    """Test MVPA analysis components"""
    
    @patch('mvpa_analysis.MVPAAnalysis.create_maskers')
    def test_mvpa_initialization(self, mock_create_maskers, mock_config):
        """Test MVPA analysis initialization"""
        from mvpa_analysis import MVPAAnalysis
        
        # Mock masker creation
        mock_create_maskers.return_value = {'test_roi': Mock()}
        
        analysis = MVPAAnalysis(config=mock_config)
        
        assert analysis.name == 'MVPAAnalysis'
        assert analysis.config == mock_config
        assert 'algorithms' in analysis.mvpa_params
        assert 'classification' in analysis.mvpa_params['algorithms']
        assert 'regression' in analysis.mvpa_params['algorithms']
    
    @patch('mvpa_analysis.extract_neural_patterns')
    def test_extract_trial_data(self, mock_extract, mock_config, synthetic_fmri_data, synthetic_behavioral_data):
        """Test trial data extraction"""
        from mvpa_analysis import MVPAAnalysis
        
        # Mock pattern extraction
        mock_extract.return_value = {
            'success': True,
            'patterns': np.random.randn(N_TRIALS, N_VOXELS)
        }
        
        analysis = MVPAAnalysis(config=mock_config)
        analysis.maskers = {'test_roi': Mock()}
        
        X = analysis.extract_trial_data(
            synthetic_fmri_data, synthetic_behavioral_data, 'test_roi'
        )
        
        assert X.shape == (N_TRIALS, N_VOXELS)
        mock_extract.assert_called_once()
    
    @patch('mvpa_analysis.extract_neural_patterns')
    def test_extract_trial_data_failure(self, mock_extract, mock_config, synthetic_fmri_data, synthetic_behavioral_data):
        """Test trial data extraction failure"""
        from mvpa_analysis import MVPAAnalysis
        from analysis_base import AnalysisError
        
        # Mock pattern extraction failure
        mock_extract.return_value = {
            'success': False,
            'error': 'Pattern extraction failed'
        }
        
        analysis = MVPAAnalysis(config=mock_config)
        analysis.maskers = {'test_roi': Mock()}
        
        with pytest.raises(AnalysisError, match="Pattern extraction failed"):
            analysis.extract_trial_data(
                synthetic_fmri_data, synthetic_behavioral_data, 'test_roi'
            )
    
    @patch('mvpa_analysis.run_classification')
    def test_decode_choices(self, mock_classify, mock_config, synthetic_neural_patterns):
        """Test choice decoding"""
        from mvpa_analysis import MVPAAnalysis
        
        # Mock classification
        mock_classify.return_value = {
            'success': True,
            'accuracy': 0.65,
            'p_value': 0.02
        }
        
        analysis = MVPAAnalysis(config=mock_config)
        
        X = synthetic_neural_patterns
        y = np.random.randint(0, 2, N_TRIALS)
        
        result = analysis.decode_choices(X, y, 'test_roi')
        
        assert result['success'] == True
        assert result['accuracy'] == 0.65
        assert result['roi_name'] == 'test_roi'
        assert result['analysis_type'] == 'choice_decoding'
    
    @patch('mvpa_analysis.run_regression')
    def test_decode_continuous_variable(self, mock_regress, mock_config, synthetic_neural_patterns):
        """Test continuous variable decoding"""
        from mvpa_analysis import MVPAAnalysis
        
        # Mock regression
        mock_regress.return_value = {
            'success': True,
            'score': 0.25,
            'p_value': 0.04
        }
        
        analysis = MVPAAnalysis(config=mock_config)
        
        X = synthetic_neural_patterns
        y = np.random.randn(N_TRIALS)
        
        result = analysis.decode_continuous_variable(X, y, 'test_roi', 'sv_diff')
        
        assert result['success'] == True
        assert result['score'] == 0.25
        assert result['roi_name'] == 'test_roi'
        assert result['variable_name'] == 'sv_diff'
        assert result['analysis_type'] == 'continuous_decoding'
    
    @patch('mvpa_analysis.MVPAAnalysis.load_behavioral_data')
    @patch('mvpa_analysis.MVPAAnalysis.load_fmri_data')
    @patch('mvpa_analysis.MVPAAnalysis.extract_trial_data')
    def test_process_subject_success(self, mock_extract, mock_load_fmri, mock_load_behavior,
                                   mock_config, synthetic_behavioral_data, synthetic_fmri_data,
                                   synthetic_neural_patterns):
        """Test successful MVPA subject processing"""
        from mvpa_analysis import MVPAAnalysis
        
        # Mock data loading
        mock_load_behavior.return_value = synthetic_behavioral_data
        mock_load_fmri.return_value = (synthetic_fmri_data, None)
        mock_extract.return_value = synthetic_neural_patterns
        
        analysis = MVPAAnalysis(config=mock_config)
        analysis.maskers = {'test_roi': Mock()}
        
        with patch.object(analysis, 'decode_choices') as mock_decode_choices, \
             patch.object(analysis, 'decode_continuous_variable') as mock_decode_continuous:
            
            mock_decode_choices.return_value = {'success': True, 'accuracy': 0.65}
            mock_decode_continuous.return_value = {'success': True, 'score': 0.25}
            
            result = analysis.process_subject('test_subject_001')
            
            assert result['success'] == True
            assert 'roi_results' in result
            assert result['subject_id'] == 'test_subject_001'
            assert result['n_successful_rois'] >= 0
    
    def test_create_summary_dataframe(self, mock_config):
        """Test MVPA summary dataframe creation"""
        from mvpa_analysis import MVPAAnalysis
        
        analysis = MVPAAnalysis(config=mock_config)
        
        # Add some fake results
        analysis.results = {
            'subject_001': {
                'success': True,
                'roi_results': {
                    'test_roi': {
                        'success': True,
                        'choice_decoding': {
                            'success': True,
                            'accuracy': 0.65,
                            'p_value': 0.02
                        },
                        'continuous_decoding': {
                            'sv_diff': {
                                'success': True,
                                'score': 0.25,
                                'p_value': 0.04
                            }
                        },
                        'n_trials': 100,
                        'n_voxels': 500
                    }
                }
            }
        }
        
        df = analysis.create_mvpa_summary_dataframe()
        
        assert len(df) == 2  # Choice + continuous decoding
        assert 'subject_id' in df.columns
        assert 'roi_name' in df.columns
        assert 'analysis_type' in df.columns
        assert 'score' in df.columns


# ============================================================================
# Test Geometry Analysis
# ============================================================================

class TestGeometryAnalysis:
    """Test geometry analysis components"""
    
    def test_geometry_initialization(self, mock_config):
        """Test geometry analysis initialization"""
        from geometry_analysis import GeometryAnalysis
        
        analysis = GeometryAnalysis(config=mock_config)
        
        assert analysis.name == 'GeometryAnalysis'
        assert analysis.config == mock_config
        assert 'dimensionality_reduction' in analysis.geometry_params
        assert 'rdm' in analysis.geometry_params
        assert 'visualization' in analysis.geometry_params
    
    def test_compute_neural_rdm(self, mock_config, synthetic_neural_patterns):
        """Test RDM computation"""
        from geometry_analysis import GeometryAnalysis
        
        analysis = GeometryAnalysis(config=mock_config)
        
        X = synthetic_neural_patterns
        rdm = analysis.compute_neural_rdm(X)
        
        assert rdm.shape == (N_TRIALS, N_TRIALS)
        assert np.allclose(rdm, rdm.T)  # Should be symmetric
        assert np.allclose(np.diag(rdm), 0)  # Diagonal should be zero
    
    @patch('geometry_analysis.run_dimensionality_reduction')
    def test_dimensionality_reduction(self, mock_dimred, mock_config, synthetic_neural_patterns):
        """Test dimensionality reduction"""
        from geometry_analysis import GeometryAnalysis
        
        # Mock dimensionality reduction
        embedding = np.random.randn(N_TRIALS, 10)
        mock_dimred.return_value = {
            'success': True,
            'embedding': embedding,
            'reducer': Mock()
        }
        
        analysis = GeometryAnalysis(config=mock_config)
        
        X = synthetic_neural_patterns
        result_embedding, reducer = analysis.dimensionality_reduction(X)
        
        assert result_embedding.shape == (N_TRIALS, 10)
        mock_dimred.assert_called_once()
    
    def test_behavioral_geometry_correlation(self, mock_config):
        """Test behavioral-geometry correlations"""
        from geometry_analysis import GeometryAnalysis
        
        analysis = GeometryAnalysis(config=mock_config)
        
        # Create synthetic embedding and behavioral data
        embedding = np.random.randn(N_TRIALS, 5)
        behavioral_vars = {
            'sv_diff': np.random.randn(N_TRIALS),
            'choice': np.random.randint(0, 2, N_TRIALS)
        }
        
        correlations = analysis.behavioral_geometry_correlation(embedding, behavioral_vars)
        
        assert 'sv_diff' in correlations
        assert 'choice' in correlations
        assert 'correlations' in correlations['sv_diff']
        assert 'max_correlation' in correlations['sv_diff']
        assert len(correlations['sv_diff']['correlations']) == 5
    
    def test_compare_embeddings_by_condition(self, mock_config):
        """Test condition-based embedding comparisons"""
        from geometry_analysis import GeometryAnalysis
        
        analysis = GeometryAnalysis(config=mock_config)
        
        # Create synthetic embedding with clear condition differences
        embedding = np.random.randn(N_TRIALS, 5)
        conditions = np.random.randint(0, 2, N_TRIALS)
        
        # Add some structure to make conditions more distinguishable
        embedding[conditions == 0] += 1
        embedding[conditions == 1] -= 1
        
        comparison_results = analysis.compare_embeddings_by_condition(
            embedding, conditions, n_permutations=100
        )
        
        assert 'condition_names' in comparison_results
        assert 'properties' in comparison_results
        assert 'centroid_distance' in comparison_results['properties']
        assert 'within_condition_variance' in comparison_results['properties']
        assert 'between_condition_variance' in comparison_results['properties']
        
        # Check that each property has required fields
        for prop_name, prop_result in comparison_results['properties'].items():
            assert 'observed' in prop_result
            assert 'p_value' in prop_result
            assert 'z_score' in prop_result
    
    @patch('geometry_analysis.GeometryAnalysis.visualize_embeddings')
    def test_visualize_embeddings(self, mock_visualize, mock_config):
        """Test embedding visualization"""
        from geometry_analysis import GeometryAnalysis
        
        # Mock visualization
        mock_visualize.return_value = {
            '2d_projections': '/path/to/2d_plot.png',
            '3d_embedding': '/path/to/3d_plot.png'
        }
        
        analysis = GeometryAnalysis(config=mock_config)
        
        embedding = np.random.randn(N_TRIALS, 5)
        behavioral_vars = {'choice': np.random.randint(0, 2, N_TRIALS)}
        
        figures = analysis.visualize_embeddings(embedding, behavioral_vars, 'test_roi')
        
        assert '2d_projections' in figures
        assert '3d_embedding' in figures
        mock_visualize.assert_called_once()


# ============================================================================
# Test Mask Creation and Validation
# ============================================================================

class TestMaskCreation:
    """Test mask creation and validation"""
    
    @patch('analysis_base.check_mask_exists')
    @patch('analysis_base.load_mask')
    def test_create_maskers_success(self, mock_load_mask, mock_check_exists, mock_config):
        """Test successful masker creation"""
        from analysis_base import BaseAnalysis
        from behavioral_analysis import BehavioralAnalysis
        
        # Mock successful mask loading
        mock_check_exists.return_value = True
        mock_load_mask.return_value = Mock()
        
        analysis = BehavioralAnalysis(config=mock_config)
        maskers = analysis.create_maskers(['test_roi_1', 'test_roi_2'])
        
        assert 'test_roi_1' in maskers
        assert 'test_roi_2' in maskers
        assert len(maskers) == 2
    
    @patch('analysis_base.check_mask_exists')
    def test_create_maskers_missing_mask(self, mock_check_exists, mock_config):
        """Test masker creation with missing mask"""
        from behavioral_analysis import BehavioralAnalysis
        
        # Mock missing mask
        mock_check_exists.return_value = False
        
        analysis = BehavioralAnalysis(config=mock_config)
        maskers = analysis.create_maskers(['test_roi_1', 'test_roi_2'])
        
        # Should return empty dict when masks don't exist
        assert len(maskers) == 0
    
    @patch('analysis_base.check_mask_exists')
    @patch('analysis_base.load_mask')
    def test_create_maskers_load_failure(self, mock_load_mask, mock_check_exists, mock_config):
        """Test masker creation with load failure"""
        from behavioral_analysis import BehavioralAnalysis
        
        # Mock mask exists but loading fails
        mock_check_exists.return_value = True
        mock_load_mask.side_effect = Exception("Load failed")
        
        analysis = BehavioralAnalysis(config=mock_config)
        maskers = analysis.create_maskers(['test_roi_1'])
        
        # Should handle the error gracefully
        assert len(maskers) == 0


# ============================================================================
# Test Data Loading and Utilities
# ============================================================================

class TestDataLoading:
    """Test data loading utilities"""
    
    @patch('analysis_base.load_behavioral_data')
    def test_load_behavioral_data_success(self, mock_load_data, mock_config, synthetic_behavioral_data):
        """Test successful behavioral data loading"""
        from behavioral_analysis import BehavioralAnalysis
        
        # Mock data loading
        mock_load_data.return_value = synthetic_behavioral_data
        
        analysis = BehavioralAnalysis(config=mock_config)
        data = analysis.load_behavioral_data('test_subject_001')
        
        assert len(data) == N_TRIALS
        assert 'choice' in data.columns
        assert 'test_subject_001' in analysis._data_cache['behavioral_test_subject_001']
    
    @patch('analysis_base.load_behavioral_data')
    def test_load_behavioral_data_failure(self, mock_load_data, mock_config):
        """Test behavioral data loading failure"""
        from behavioral_analysis import BehavioralAnalysis
        from data_utils import DataError
        from analysis_base import AnalysisError
        
        # Mock data loading failure
        mock_load_data.side_effect = DataError("Data file not found")
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        with pytest.raises(AnalysisError, match="Behavioral data loading failed"):
            analysis.load_behavioral_data('test_subject_001')
    
    @patch('analysis_base.load_fmri_data')
    @patch('analysis_base.load_confounds')
    def test_load_fmri_data_success(self, mock_load_confounds, mock_load_fmri, mock_config, 
                                   synthetic_fmri_data, synthetic_confounds):
        """Test successful fMRI data loading"""
        from behavioral_analysis import BehavioralAnalysis
        
        # Mock data loading
        mock_load_fmri.return_value = synthetic_fmri_data
        mock_load_confounds.return_value = synthetic_confounds
        
        analysis = BehavioralAnalysis(config=mock_config)
        img, confounds = analysis.load_fmri_data('test_subject_001')
        
        assert img.shape == synthetic_fmri_data.shape
        assert confounds.shape == synthetic_confounds.shape
    
    @patch('analysis_base.get_complete_subjects')
    def test_get_subject_list(self, mock_get_subjects, mock_config):
        """Test subject list retrieval"""
        from behavioral_analysis import BehavioralAnalysis
        
        # Mock subject list
        mock_get_subjects.return_value = TEST_SUBJECTS
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Test with no specific subjects
        subjects = analysis.get_subject_list()
        assert subjects == TEST_SUBJECTS
        
        # Test with specific subjects
        specific_subjects = ['subject_001']
        subjects = analysis.get_subject_list(specific_subjects)
        assert subjects == specific_subjects


# ============================================================================
# Test Error Handling and Edge Cases
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_analysis_error_inheritance(self):
        """Test AnalysisError inheritance"""
        from analysis_base import AnalysisError
        
        error = AnalysisError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_empty_data_handling(self, mock_config):
        """Test handling of empty data"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        validation_result = analysis.validate_behavioral_data(empty_data)
        
        assert validation_result['valid'] == False
        assert 'Insufficient trials' in validation_result['errors'][0]
    
    def test_invalid_roi_name(self, mock_config):
        """Test handling of invalid ROI names"""
        from mvpa_analysis import MVPAAnalysis
        from analysis_base import AnalysisError
        
        analysis = MVPAAnalysis(config=mock_config)
        
        with pytest.raises(AnalysisError, match="Masker for invalid_roi not found"):
            analysis.extract_trial_data(Mock(), Mock(), 'invalid_roi')
    
    def test_processing_stats_update(self, mock_config):
        """Test processing statistics updates"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Test successful processing
        analysis.update_processing_stats('subject_001', 1.5, success=True, memory_usage=150.0)
        
        assert analysis.processing_stats['subjects_processed'] == 1
        assert analysis.processing_stats['subjects_failed'] == 0
        assert analysis.processing_stats['processing_times'] == [1.5]
        assert analysis.processing_stats['memory_usage'] == [150.0]
        
        # Test failed processing
        analysis.update_processing_stats('subject_002', 0.8, success=False, memory_usage=100.0)
        
        assert analysis.processing_stats['subjects_processed'] == 1
        assert analysis.processing_stats['subjects_failed'] == 1
        assert len(analysis.processing_stats['processing_times']) == 2
        assert len(analysis.processing_stats['memory_usage']) == 2


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    @patch('behavioral_analysis.BehavioralAnalysis.load_behavioral_data')
    def test_full_behavioral_pipeline(self, mock_load_data, mock_config, synthetic_behavioral_data):
        """Test complete behavioral analysis pipeline"""
        from behavioral_analysis import BehavioralAnalysis
        
        # Mock data loading
        mock_load_data.return_value = synthetic_behavioral_data
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Run analysis on test subjects
        results = analysis.run_analysis(['test_subject_001', 'test_subject_002'])
        
        assert 'results' in results
        assert 'summary' in results
        assert 'successful_subjects' in results
        assert 'failed_subjects' in results
        
        # Check summary stats
        summary = results['summary']
        assert 'n_subjects_total' in summary
        assert 'success_rate' in summary
        
        # Check individual results
        for subject_id in ['test_subject_001', 'test_subject_002']:
            assert subject_id in results['results']
            result = results['results'][subject_id]
            assert result['success'] == True
            assert 'k' in result
            assert 'processing_time' in result
    
    def test_analysis_factory_integration(self, mock_config):
        """Test analysis factory integration"""
        from analysis_base import AnalysisFactory
        
        # Create all analysis types
        analyses = {}
        for analysis_type in ['behavioral', 'mvpa', 'geometry']:
            analyses[analysis_type] = AnalysisFactory.create(analysis_type, config=mock_config)
        
        # Test that all have common interface
        for analysis_type, analysis in analyses.items():
            assert hasattr(analysis, 'process_subject')
            assert hasattr(analysis, 'run_analysis')
            assert hasattr(analysis, 'save_results')
            assert hasattr(analysis, 'load_results')
            assert hasattr(analysis, 'get_analysis_summary')
    
    def test_memory_efficient_integration(self, mock_config):
        """Test memory efficient integration"""
        from analysis_base import AnalysisFactory
        
        try:
            from memory_efficient_data import MemoryConfig
            
            # Create memory configuration
            memory_config = MemoryConfig()
            
            # Test creation with memory efficiency
            analysis = AnalysisFactory.create(
                'behavioral',
                config=mock_config,
                enable_memory_efficient=True,
                memory_config=memory_config
            )
            
            assert analysis.enable_memory_efficient == True
            assert analysis.memory_config is not None
            
        except ImportError:
            # Memory efficient utilities not available
            pytest.skip("Memory efficient utilities not available")


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for the pipeline"""
    
    def test_behavioral_analysis_performance(self, mock_config):
        """Test behavioral analysis performance"""
        from behavioral_analysis import BehavioralAnalysis
        import time
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Create larger dataset
        n_trials = 1000
        choices = np.random.randint(0, 2, n_trials)
        amounts = np.random.uniform(25, 100, n_trials)
        delays = np.random.uniform(1, 365, n_trials)
        
        start_time = time.time()
        fit_result = analysis.fit_discount_rate(choices, amounts, delays)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # 5 seconds
        assert fit_result['success'] == True
    
    def test_cache_performance(self, mock_config):
        """Test cache performance"""
        from behavioral_analysis import BehavioralAnalysis
        
        analysis = BehavioralAnalysis(config=mock_config)
        
        # Add large amount of data to cache
        large_data = np.random.randn(1000, 1000)
        analysis._data_cache['large_data'] = large_data
        
        # Test cache info
        cache_info = analysis.get_cache_info()
        assert cache_info['cache_size'] == 1
        assert cache_info['memory_usage_mb'] > 0
        
        # Test cache clearing
        analysis.clear_cache()
        assert len(analysis._data_cache) == 0


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"]) 