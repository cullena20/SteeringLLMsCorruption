"""
Shared estimator definitions, per-model alpha values, and default settings used across
experiment scripts. 
"""

from estimators.steering_only_estimators import sample_diff_of_means
from estimators.steering_estimator_wrappers import diff_of_means, mean_of_diffs
from estimators.que import que_mean
from estimators.simple_estimators import median_of_means
from estimators.lee_valiant import lee_valiant_simple
from estimators.lrv import lrv
from estimators.simple_estimators import coord_trimmed_mean

# Estimator lambdas — "diff" variants compute mean(pos) - mean(neg);
# "match" variants compute mean(pos - neg).
que_diff = lambda pos, neg, tau: diff_of_means(pos, neg, tau=tau, mean_fun=que_mean, return_outlier_indices=True)
que_match = lambda pos, neg, tau: mean_of_diffs(pos, neg, tau=tau, mean_fun=que_mean, mismatch=False, return_outlier_indices=True)
med_mean_diff = lambda pos, neg: diff_of_means(pos, neg, mean_fun=median_of_means)
med_mean_match = lambda pos, neg: mean_of_diffs(pos, neg, mean_fun=median_of_means, mismatch=False)
lee_valiant_diff = lambda pos, neg, tau: diff_of_means(pos, neg, tau=tau, mean_fun=lee_valiant_simple, return_outlier_indices=True)
lee_valiant_match = lambda pos, neg, tau: mean_of_diffs(pos, neg, tau=tau, mean_fun=lee_valiant_simple, mismatch=False, return_outlier_indices=True)
lrv_diff = lambda pos, neg: diff_of_means(pos, neg, mean_fun=lrv)
lrv_match = lambda pos, neg: mean_of_diffs(pos, neg, mean_fun=lrv, mismatch=False)
coord_prune_diff = lambda pos, neg, tau: diff_of_means(pos, neg, tau=tau, mean_fun=coord_trimmed_mean)
coord_prune_match = lambda pos, neg, tau: mean_of_diffs(pos, neg, tau=tau, mean_fun=coord_trimmed_mean)
que_diff_force_prune = lambda pos, neg, tau: diff_of_means(pos, neg, tau=0.5*tau, mean_fun=que_mean, return_outlier_indices=True, always_prune=True)

# Mapping from CLI name → estimator function, used by --estimator-names.
ESTIMATOR_MAPPING = {
    "que_diff": que_diff,
    "que_match": que_match,
    "que_diff_force_prune": que_diff_force_prune, # variant of que_diff that forces pruning at 0.5*tau, included for ablation purposes, doesn't work
    "med_mean_diff": med_mean_diff,
    "med_mean_match": med_mean_match,
    "lee_valiant_diff": lee_valiant_diff,
    "lee_valiant_match": lee_valiant_match,
    "lrv_diff": lrv_diff,
    "lrv_match": lrv_match,
    "coord_prune_diff": coord_prune_diff,
    "coord_prune_match": coord_prune_match,
    "sample_diff_of_means": sample_diff_of_means,
}

# per behavior optimal alpha values for each model 

LLAMA_ALPHA_VALUES = {
    "uncorrigible-neutral-HHH": 0.75,
    "myopic-reward": 0.75,
    "power-seeking-inclination": 1,
    "wealth-seeking-inclination": 1,
    "survival-instinct": 0.75,
    "coordinate-other-ais": 1,
}

OLMO_ALPHA_VALUES = {
    "uncorrigible-neutral-HHH": 1,
    "myopic-reward": 0.75,
    "power-seeking-inclination": 0.75,
    "wealth-seeking-inclination": 1,
    "survival-instinct": 0.75,
    "coordinate-other-ais": 0.75,
}

MISTRAL_ALPHA_VALUES = {
    "uncorrigible-neutral-HHH": 1,
    "myopic-reward": 0.5,
    #"self-awareness-text-model": 0.25,
    "power-seeking-inclination": 1,
    "wealth-seeking-inclination": 1,
    "survival-instinct": 0.75,
    "coordinate-other-ais": 0.5,
}

DEFAULT_BEHAVIORS = [
    "uncorrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
    "wealth-seeking-inclination",
    "survival-instinct",
    "coordinate-other-ais",
]
