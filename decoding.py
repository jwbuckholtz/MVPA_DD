
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, permutation_test_score
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, Lasso
import config

def classify(X, y, n_splits=5, random_state=42):
    """Performs classification using an SVM."""
    clf = SVC(kernel=config.MVPA.CLASSIFIER, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    score, _, p_value = permutation_test_score(clf, X, y, cv=cv, n_permutations=config.MVPA.N_PERMUTATIONS, n_jobs=config.System.N_JOBS)
    return {'score': score, 'p_value': p_value}

def regress(X, y, n_splits=5, random_state=42):
    """Performs regression using Ridge."""
    reg = Ridge(alpha=1.0)
    if config.MVPA.REGRESSOR == 'lasso':
        reg = Lasso(alpha=1.0)
    
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    score, _, p_value = permutation_test_score(reg, X, y, cv=cv, n_permutations=config.MVPA.N_PERMUTATIONS, n_jobs=config.System.N_JOBS)
    return {'score': score, 'p_value': p_value} 