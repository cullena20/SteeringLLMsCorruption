"""
Mislabelling corruption experiment.

Corrupts the training activations by swapping positive and negative labels for an eta fraction
of the data, equivalent to swapping pos/neg activations. Runs the full steering pipeline and
evaluates performance on the behavior test set.

Results are saved to experiment_results/mislabel/ as pickled ExperimentOutput objects, along
with steering vectors and outlier detection data. Plots of steering performance vs corruption
percentage are saved to saved_plots/mislabel/.
"""

import sys, os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.steerable_model import SteerableModel
from src.dataset import DataSet
from src.utils import set_global_seed, large_dataset_generator
from src.corruption import label_corruption
from src.corrupted_activations import CorruptedActivations
from src.eval_utils import evaluate_across_behaviors_and_vecs, grab_generations_across_behaviors_and_vecs, tinymmlu_eval
from src.experiment_output import ExperimentOutput
from experiments.config import (
    ESTIMATOR_MAPPING, LLAMA_ALPHA_VALUES, OLMO_ALPHA_VALUES, MISTRAL_ALPHA_VALUES,
    DEFAULT_BEHAVIORS, sample_diff_of_means, lee_valiant_diff,
)

import time
import pickle
from datasets import load_dataset

set_global_seed(42)


def main(
    model_path: str,
    activations_name: str, # ASSUMED TO NOT CONTAIN .pkl
    alpha: float, # possibly different value for different models
    layer: int, # possibly different value for different models, zero indexed
    batch_size: int = 16,
    estimator_names: list | None = None,
    etas: list | None = None,
    behaviors: list | None = None,
    test_size: int = 200,
    grab_generations: bool = False,
    tiny_mmlu: bool = False,
    save_steering_vecs: bool = True,
    runs: int = 3,
    save_name_postfix: str = "", # in case we want to run multiple variants
    use_large_dataset: bool = False, # specific to some datasets for a particular experiment, not general
    custom_dataset_path: str | None = None, # if provided, overrides other dataset loading logic and loads from this path instead (expects same format as DataSet class)
    supply_steering_path: str | None = None, # added for precomputed steering vecs use -> needed for robust mean estimation suite, where steering vec calculation is the bottleneck
    grab_steering_vecs_only: bool = False, # if true, only grab steering vecs and save them, do not do evaluations -> needed for robust mean estimation suite
):
    if grab_generations:
        dataset = DataSet(subfolders=["open_ended"], test_size=test_size, format_type="open_ended")
    elif tiny_mmlu:
        dataset = load_dataset("tinyBenchmarks/tinyMMLU")["test"]
    elif custom_dataset_path is not None:
        try:
            dataset = DataSet(subfolders=[custom_dataset_path], test_size=test_size)
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
    elif use_large_dataset:
        dataset = large_dataset_generator(test_size=test_size)
    else:
        dataset = DataSet(subfolders=["tan_paper_datasets/mwe/xrisk"], test_size=test_size)

    if behaviors is None:
        behaviors = list(DEFAULT_BEHAVIORS)
    if etas is None:
        etas = [0, 0.1, 0.2, 0.3, 0.4]

    if estimator_names is None:
        estimators = {
            "sample_diff_of_means": sample_diff_of_means,
            "lee_valiant_diff": lee_valiant_diff
        }
    else:
        estimators = {name: ESTIMATOR_MAPPING[name] for name in estimator_names}

    activations_base_path = f"{PROJECT_ROOT}/activations"
    activations_path = f"{activations_base_path}/{activations_name}.pkl"

    with open(activations_path, "rb") as f:
        activations_obj = pickle.load(f)

    corrupted_activations = CorruptedActivations(activations_obj)

    if not grab_steering_vecs_only:
        model = SteerableModel(model_name=model_path)
        print("Successfully loaded model")
    else:
        model = None
        print("Skipping model load for grab steering vectors only run.")

    if supply_steering_path is not None:
        with open(f"{PROJECT_ROOT}/{supply_steering_path}", "rb") as f:
            supply_steering_vecs = pickle.load(f)
        try:
            supply_steering_vecs = supply_steering_vecs["steering_vecs"]
        except:
            pass
        print("Loaded precomputed steering vectors from supplied path.")
    else:
        supply_steering_vecs = None

    seed_run_kwargs = {i: {"seed": i} for i in range(runs)}
    # Corrupt data over 3 runs for each eta (runs correspond to random seeds)
    for eta in etas:
        corrupted_activations.apply_corruption(
            corruption_fun=label_corruption,
            eta=eta,
            corruption_name=f"label_corruption_eta_{eta}",
            behaviors=behaviors,
            token_positions=["answer_token"],
            multiple_runs_kwargs=seed_run_kwargs
        )
    print("Completed corruption with mislabel corruption")

    intervention_layers = [layer]
    alpha_values = [alpha]

    experiment = ExperimentOutput(generation_mode=grab_generations, benchmark_mode=tiny_mmlu)

    first_run = True

    steering_vecs_full = {}
    outliers_detected_data_full = {}

    # Override alpha with behavior specific
    if "llama" in model_path.lower():
        alpha_mapping = LLAMA_ALPHA_VALUES
    elif "olmo" in model_path.lower():
        alpha_mapping = OLMO_ALPHA_VALUES
    elif "mistral" in model_path.lower():
        alpha_mapping = MISTRAL_ALPHA_VALUES
    else:
        alpha_mapping = None

    if grab_generations:
        printing_x = "generation"
    else:
        printing_x = "evaluation"

    print("Beginning evaluation across corrupted activations.")
    time_start_total = time.time()
    for corruption_name, corrupted_version in corrupted_activations.corrupted_versions.items():
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

            corrupted_activations = corrupted_version[run] # stores runs

            if supply_steering_vecs is None:
                steering_vecs, outliers_detected_data = corrupted_activations.get_steering_vecs(
                    estimators,
                    behaviors,
                    eta=float(eta),
                    include_sample_uncorrupted=True,
                    return_outlier_info=True,
                )
            else:
                steering_vecs = supply_steering_vecs[eta][run] # precomputed steering vecs
                outliers_detected_data = None

            steering_vecs_full[eta][run] = steering_vecs
            outliers_detected_data_full[eta][run] = outliers_detected_data

            if grab_generations:
                grab_generations_across_behaviors_and_vecs(
                    model,
                    steering_vecs,
                    dataset.test_data,
                    intervention_layers=intervention_layers,
                    alpha_values=alpha_values, # overriden by behavior specific alpha values
                    generation_batch_size=batch_size,
                    include_no_steer=include_no_steer,
                    experiment_output=experiment, # accumulate results in this single object
                    varying_variable="eta",
                    varying_variable_value=eta,
                    run=run,
                    behavior_alpha_mapping=alpha_mapping,
                )
            elif tiny_mmlu:
                tinymmlu_eval(
                    model,
                    steering_vecs,
                    dataset, # dataset here of different format
                    intervention_layers=intervention_layers,
                    alpha_values=alpha_values, # overriden by behavior specific alpha values
                    generation_batch_size=batch_size,
                    include_no_steer=include_no_steer,
                    experiment_output=experiment, # accumulate results in this single object
                    varying_variable="eta",
                    varying_variable_value=eta,
                    run=run,
                    behavior_alpha_mapping=alpha_mapping,
                )
            elif not grab_steering_vecs_only:
                evaluate_across_behaviors_and_vecs(
                    model,
                    steering_vecs,
                    dataset.test_data,
                    intervention_layers=intervention_layers,
                    alpha_values=alpha_values,
                    generation_batch_size=batch_size,
                    include_no_steer=include_no_steer,
                    experiment_output=experiment, # accumulate results in this single object
                    varying_variable="eta",
                    varying_variable_value=eta,
                    run=run,
                    behavior_alpha_mapping=alpha_mapping,
                )
            time_end = time.time()
            print(f"Completed {printing_x} for {corruption_name}, Run {run}, in {time_end - time_start} seconds")
    time_end_total = time.time()
    print(f"Completed all {printing_x}s in {time_end_total - time_start_total} seconds")

    if grab_generations:
        save_path = f"{PROJECT_ROOT}/experiment_results/mislabel/{activations_name}_{save_name_postfix}_generations.pkl"
    elif tiny_mmlu:
        save_path = f"{PROJECT_ROOT}/experiment_results/mislabel/{activations_name}_{save_name_postfix}_tinymmlu_eval.pkl"
    else:
        save_path = f"{PROJECT_ROOT}/experiment_results/mislabel/{activations_name}_{save_name_postfix}.pkl"

    if not grab_steering_vecs_only:
        with open(save_path, "wb") as f:
            pickle.dump(experiment, f)

    print("Saved experiment results")

    steering_vec_info = {
        "steering_vecs": steering_vecs_full,
        "outliers_detected_data": outliers_detected_data_full
    }

    with open(f"{PROJECT_ROOT}/experiment_results/mislabel/{activations_name}_steering_vecs_and_outlier_detected_data_{save_name_postfix}.pkl", "wb") as f:
        pickle.dump(steering_vec_info, f)

    print("Saved steering vector and outlier detected data info")

    if not grab_generations and not tiny_mmlu and not grab_steering_vecs_only:
        try:
            experiment.plot_performance_grid(
                xlabel="Corruption Percentage",
                metric="percent_steered",
                save_title=f"{PROJECT_ROOT}/saved_plots/mislabel/{activations_name}_percent-steered_{save_name_postfix}"
            )
            experiment.plot_performance_grid(
                xlabel="Corruption Percentage",
                metric="avg_score",
                save_title=f"{PROJECT_ROOT}/saved_plots/mislabel/{activations_name}_avg-score_{save_name_postfix}"
            )
            print("Saved performance grid plot")
        except Exception as e:
            print(f"Could not save performance grid plot due to error: {e}")
    elif tiny_mmlu:
        try:
            experiment.plot_performance_grid(
                xlabel="Corruption Percentage",
                metric="score",
                save_title=f"{PROJECT_ROOT}/saved_plots/mislabel/{activations_name}_mmlu-accuracy_{save_name_postfix}"
            )
            print("Saved MMLU accuracy grid plot")
        except Exception as e:
            print(f"Could not save MMLU accuracy grid plot due to error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mislabel corruption steerability experiments"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="HuggingFace model path or local checkpoint"
    )

    parser.add_argument(
        "--activations_name",
        type=str,
        help="Name of the activations file"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Alpha value for steering evaluation"
    )

    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index to evaluate"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )

    parser.add_argument(
        "--estimator-names",
        type=str,
        nargs="+",
        default=None,
        help="List of estimator names to use (space-separated). Defaults to sample_diff_of_means and lee_valiant_diff."
    )

    parser.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=None,
        help="List of eta values for mislabel corruption (space-separated). Defaults to [0, 0.1, 0.2, 0.3, 0.4]."
    )

    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=None,
        help="List of behaviors to evaluate (space-separated). Defaults to default 6 behaviors set."
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Test set size (default: 200)"
    )

    parser.add_argument(
        "--grab-generations",
        action="store_true",
        help="Flag to indicate whether to grab generations instead of evaluations"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs (default: 3)"
    )

    parser.add_argument(
        "--save-name-postfix",
        type=str,
        default="",
        help="Postfix to add to saved experiment result filenames"
    )

    parser.add_argument(
        "--tiny-mmlu",
        action="store_true",
        help="Flag to indicate whether to use tiny mmlu benchmarking instead of behavior evaluation"
    )

    parser.add_argument(
        "--use-large-dataset",
        action="store_true",
        help="Whether to use the large datasets specific to certain experiments"
    )

    parser.add_argument(
        "--custom-dataset-path",
        type=str,
        default=None,
        help="Path to a custom dataset folder. If provided, overrides the default dataset path."
    )

    parser.add_argument(
        "--grab-steering-vecs-only",
        action="store_true",
        help="If true, only grab steering vecs and save them, do not do evaluations"
    )

    parser.add_argument(
        "--supply-steering-path",
        type=str,
        default=None,
        help="Path to precomputed steering vectors to use"
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
        behaviors=args.behaviors,
        test_size=args.test_size,
        grab_generations=args.grab_generations,
        runs=args.runs,
        save_name_postfix=args.save_name_postfix,
        tiny_mmlu=args.tiny_mmlu,
        use_large_dataset=args.use_large_dataset,
        custom_dataset_path=args.custom_dataset_path,
        grab_steering_vecs_only=args.grab_steering_vecs_only,
        supply_steering_path=args.supply_steering_path,
    )
