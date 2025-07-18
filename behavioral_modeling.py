
import numpy as np
from scipy import optimize

def hyperbolic_discount_function(delay: np.ndarray, k: float) -> np.ndarray:
    """Hyperbolic discounting function: V = 1 / (1 + k * delay)"""
    return 1 / (1 + k * delay)

def subjective_value(amount: np.ndarray, delay: np.ndarray, k: float) -> np.ndarray:
    """Calculate subjective value given amount, delay, and discount rate"""
    return amount * hyperbolic_discount_function(delay, k)

def fit_discount_rate(choices: np.ndarray, large_amounts: np.ndarray, delays: np.ndarray, small_amount: float = 20.0):
    """Fit hyperbolic discount rate to choice data using logistic regression"""
    
    def neg_log_likelihood(k):
        if k <= 0:
            return np.inf
        
        sv_large = subjective_value(large_amounts, delays, k)
        sv_diff = sv_large - small_amount
        choice_prob = 1 / (1 + np.exp(-sv_diff))
        choice_prob = np.clip(choice_prob, 1e-15, 1 - 1e-15)
        
        nll = -np.sum(choices * np.log(choice_prob) + (1 - choices) * np.log(1 - choice_prob))
        return nll

    try:
        result = optimize.minimize_scalar(neg_log_likelihood, bounds=(1e-6, 1.0), method='bounded')
        k_fit = result.x
        nll_fit = result.fun
        
        sv_large_fit = subjective_value(large_amounts, delays, k_fit)
        sv_diff_fit = sv_large_fit - small_amount
        choice_prob_fit = 1 / (1 + np.exp(-sv_diff_fit))
        
        null_ll = -np.sum(choices * np.log(np.mean(choices)) + (1 - choices) * np.log(1 - np.mean(choices)))
        pseudo_r2 = 1 - nll_fit / null_ll

        return {
            'k': k_fit, 'nll': nll_fit, 'pseudo_r2': pseudo_r2,
            'sv_large': sv_large_fit, 'sv_small': small_amount,
            'sv_diff': sv_diff_fit, 'sv_sum': sv_large_fit + small_amount,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)} 