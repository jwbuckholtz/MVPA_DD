

class MVPAAnalysis(BaseAnalysis):
    """
    MVPA analysis class for multi-voxel pattern analysis
    
    This class inherits from BaseAnalysis and implements MVPA-specific
    functionality including neural pattern extraction, decoding, and
    cross-validation procedures.
    """
    
    def __init__(self, config: OAKConfig = None, **kwargs):
        """
        Initialize MVPA analysis
        
        Parameters:
        -----------
        config : OAKConfig, optional
            Configuration object
        **kwargs : dict
            Additional arguments for base class
        """
        super().__init__(
            config=config,
            name='MVPAAnalysis',
            **kwargs
        )
        
        # MVPA-specific settings
        self.mvpa_params = {
            'algorithms': {
                'classification': ['svm', 'logistic', 'rf'],
                'regression': ['ridge', 'lasso', 'svr']
            },
            'cv_folds': self.config.CV_FOLDS,
            'n_permutations': self.config.N_PERMUTATIONS,
            'pattern_extraction': {
                'method': 'single_timepoint',
                'hemi_lag': self.config.HEMI_LAG,
                'window_size': 3
            }
        }
        
        # Initialize maskers dictionary
        self.maskers = {}
        
        # Configure MVPA utilities
        update_mvpa_config(
            cv_folds=self.mvpa_params['cv_folds'],
            n_permutations=self.mvpa_params['n_permutations'],
            n_jobs=1  # Conservative for memory management
        )
        
        self.logger.info("MVPA analysis initialized")