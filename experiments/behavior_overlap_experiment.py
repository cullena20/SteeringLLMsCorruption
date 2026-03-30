"""
Behavior overlap corruption experiment.

Corrupts the 'cheerfulness' training activations by replacing an eta fraction with activations
from the 'cheerfulness_power_seeking' dataset — examples that simultaneously exhibit cheerfulness
and power-seeking. Evaluates whether the injected power-seeking signal is inadvertently instilled
by measuring steering performance on the 'power-seeking-inclination' test set.

Prerequisites:
  1. Generate the cheerfulness datasets (experiments/generate_cheerfulness_dataset.py)
  2. Extract activations for both behaviors (see bash commands in workflow.md or below)
  3. Run the alpha sweep on cheerfulness and update CHEERFULNESS_ALPHA below

Setup commands:
  # Extract activations for both cheerfulness behaviors
  mkdir -p experiment_results/baseline_steerability
  python experiments/get_activations.py meta-llama/Llama-3.2-3B-Instruct \\
    --custom-dataset-path cheerfulness \\
    --behaviors cheerfulness cheerfulness_power_seeking \\
    --layer 12 \\
    --save-name Llama-3.2-3B-Instruct_layer13_cheerfulness \\
    --test-size 200

  # Sweep alpha on cheerfulness evaluated on itself; inspect results to set CHEERFULNESS_ALPHA
  python experiments/evaluate_steerability_vs_alpha.py \\
    --model_path meta-llama/Llama-3.2-3B-Instruct \\
    --activations_name Llama-3.2-3B-Instruct_layer13_cheerfulness \\
    --layer 12 \\
    --custom-dataset-path cheerfulness \\
    --alphas -2 -1 0 1 2 \\
    --test-size 200

Results are saved to experiment_results/behavior_overlap/ and
plots to saved_plots/behavior_overlap/.
"""

import sys, os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import time
import pickle

from src.steerable_model import SteerableModel
from src.dataset import DataSet
from src.utils import set_global_seed
from src.corruption import corrupt_with_shared_random
from src.corrupted_activations import CorruptedActivations
from src.eval_utils import evaluate_across_behaviors_and_vecs
from src.experiment_output import ExperimentOutput
from experiments.config import (
    ESTIMATOR_MAPPING, sample_diff_of_means, lee_valiant_diff,
)

set_global_seed(42)

# Best alpha for cheerfulness steering vector found via evaluate_steerability_vs_alpha.py sweep.
# Run that script first with --alphas -2 -1 0 1 2 and update this value.
CHEERFULNESS_ALPHA = 1.0

INLIER_BEHAVIOR = "cheerfulness"
OUTLIER_SOURCE_BEHAVIOR = "cheerfulness_power_seeking"
EVAL_BEHAVIOR = "power-seeking-inclination"


def main(
    model_path: str,
    activations_name: str,  # name of pkl in activations/, without .pkl extension
    layer: int,
    alpha: float = CHEERFULNESS_ALPHA,
    batch_size: int = 16,
    estimator_names: list | None = None,
    etas: list | None = None,
    test_size: int = 200,
    runs: int = 3,
    save_steering_vecs: bool = True,
    save_name_postfix: str = "",
):
    # Evaluation dataset: power-seeking-inclination from the standard xrisk set
    dataset = DataSet(subfolders=["tan_paper_datasets/mwe/xrisk"], test_size=test_size)

    if etas is None:
        etas = [0, 0.1, 0.2, 0.3, 0.4]

    if estimator_names is None:
        estimators = {
            "sample_diff_of_means": sample_diff_of_means,
            "lee_valiant_diff": lee_valiant_diff,
        }
    else:
        estimators = {name: ESTIMATOR_MAPPING[name] for name in estimator_names}

    activations_path = f"{PROJECT_ROOT}/activations/{activations_name}.pkl"
    with open(activations_path, "rb") as f:
        activations_obj = pickle.load(f)

    # Flatten cheerfulness_power_seeking pos+neg activations into a single tensor
    # to serve as the corruption source, following the same interface as corrupt_with_shared_random
    cp_acts = activations_obj.data[OUTLIER_SOURCE_BEHAVIOR][layer]["answer_token"]
    behavior_overlap_acts = torch.cat([cp_acts["pos"], cp_acts["neg"]], dim=0)
    print(f"Loaded {behavior_overlap_acts.shape[0]} corruption-source activations from '{OUTLIER_SOURCE_BEHAVIOR}'")

    corrupted_activations_obj = CorruptedActivations(activations_obj)

    model = SteerableModel(model_name=model_path)
    print("Successfully loaded model")

    seed_run_kwargs = {i: {"seed": i} for i in range(runs)}

    for eta in etas:
        corrupted_activations_obj.apply_corruption(
            corruption_fun=corrupt_with_shared_random,
            eta=eta,
            corruption_name=f"overlap_eta_{eta}",
            behaviors=[INLIER_BEHAVIOR],
            token_positions=["answer_token"],
            multiple_runs_kwargs=seed_run_kwargs,
            random_acts=behavior_overlap_acts,
        )
    print("Completed corruption with behavior overlap activations")

    intervention_layers = [layer]
    alpha_values = [alpha]

    experiment = ExperimentOutput()
    first_run = True
    include_no_steer = True

    steering_vecs_full = {}
    outliers_detected_data_full = {}

    print("Beginning evaluation across corrupted activations.")
    time_start_total = time.time()

    for corruption_name, corrupted_version in corrupted_activations_obj.corrupted_versions.items():
        print(f"Evaluating corruption version: {corruption_name}")
        eta = corruption_name.split("_")[-1]
        steering_vecs_full[eta] = {}
        outliers_detected_data_full[eta] = {}

        time_start = time.time()

        for run in corrupted_version.keys():
            if run == "corruption_details":
                continue
            if first_run:
                include_no_steer = True
                first_run = False
            else:
                include_no_steer = False

            steering_vecs_full[eta][run] = {}
            outliers_detected_data_full[eta][run] = {}

            local_corrupted_activations = corrupted_version[run]

            steering_vecs, outliers_detected_data = local_corrupted_activations.get_steering_vecs(
                estimators,
                [INLIER_BEHAVIOR],
                eta=float(eta),
                include_sample_uncorrupted=True,
                return_outlier_info=True,
            )

            steering_vecs_full[eta][run] = steering_vecs
            outliers_detected_data_full[eta][run] = outliers_detected_data

            # Steering vec is keyed by INLIER_BEHAVIOR ('cheerfulness'); test_behaviors
            # redirects evaluation to power-seeking-inclination test data
            evaluate_across_behaviors_and_vecs(
                model,
                steering_vecs,
                dataset.test_data,
                intervention_layers=intervention_layers,
                alpha_values=alpha_values,
                generation_batch_size=batch_size,
                include_no_steer=include_no_steer,
                experiment_output=experiment,
                varying_variable="eta",
                varying_variable_value=eta,
                run=run,
                test_behaviors=[EVAL_BEHAVIOR],
                additional_info_kwargs={
                    "inlier_behavior": INLIER_BEHAVIOR,
                    "eval_behavior": EVAL_BEHAVIOR,
                },
            )

            time_end = time.time()
            print(f"Completed evaluation for {corruption_name}, Run {run}, in {time_end - time_start:.1f}s")

    time_end_total = time.time()
    print(f"Completed all evaluations in {time_end_total - time_start_total:.1f}s")

    os.makedirs(f"{PROJECT_ROOT}/experiment_results/behavior_overlap", exist_ok=True)
    save_path = f"{PROJECT_ROOT}/experiment_results/behavior_overlap/{activations_name}_{save_name_postfix}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(experiment, f)
    print("Saved experiment results")

    if save_steering_vecs:
        steering_vec_info = {
            "steering_vecs": steering_vecs_full,
            "outliers_detected_data": outliers_detected_data_full,
        }
        sv_path = f"{PROJECT_ROOT}/experiment_results/behavior_overlap/{activations_name}_steering_vecs_and_outlier_detected_data_{save_name_postfix}.pkl"
        with open(sv_path, "wb") as f:
            pickle.dump(steering_vec_info, f)
        print("Saved steering vector and outlier detected data info")

    try:
        os.makedirs(f"{PROJECT_ROOT}/saved_plots/behavior_overlap", exist_ok=True)
        experiment.plot_performance_grid(
            xlabel="Corruption Percentage",
            metric="avg_score",
            save_title=f"{PROJECT_ROOT}/saved_plots/behavior_overlap/{activations_name}_avg-score_{save_name_postfix}",
        )
        experiment.plot_performance_grid(
            xlabel="Corruption Percentage",
            metric="percent_steered",
            save_title=f"{PROJECT_ROOT}/saved_plots/behavior_overlap/{activations_name}_percent-steered_{save_name_postfix}",
        )
        print("Saved performance grid plots")
    except Exception as e:
        print(f"Could not save performance grid plots: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Behavior overlap corruption steerability experiment"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model path or local checkpoint",
    )
    parser.add_argument(
        "--activations_name",
        type=str,
        required=True,
        help="Name of the activations file (without .pkl)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=CHEERFULNESS_ALPHA,
        help=f"Alpha value for cheerfulness steering (default: {CHEERFULNESS_ALPHA}). "
             "Override after running evaluate_steerability_vs_alpha.py.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index (zero-indexed)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--estimator-names",
        type=str,
        nargs="+",
        default=None,
        help="Estimators to use. Defaults to sample_diff_of_means and lee_valiant_diff.",
    )
    parser.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=None,
        help="Corruption fractions to sweep (default: [0, 0.1, 0.2, 0.3, 0.4])",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Test set size (default: 200)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of random-seed runs per eta (default: 3)",
    )
    parser.add_argument(
        "--save-steering-vecs",
        action="store_true",
        help="Save steering vectors and outlier detection data",
    )
    parser.add_argument(
        "--save-name-postfix",
        type=str,
        default="",
        help="Postfix appended to saved filenames",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model_path,
        activations_name=args.activations_name,
        alpha=args.alpha,
        layer=args.layer,
        batch_size=args.batch_size,
        estimator_names=args.estimator_names,
        etas=args.etas,
        test_size=args.test_size,
        runs=args.runs,
        save_steering_vecs=args.save_steering_vecs,
        save_name_postfix=args.save_name_postfix,
    )
