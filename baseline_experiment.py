import argparse
from pathlib import Path

import pandas as pd
import torch

from data_preprocessing import load_clean_datasets
from data_splitting import create_fold_assignments, prepare_run
from neural_network import ExpandableNeuralNetwork
from training_utils import (
    evaluate_model,
    save_classification_outputs,
    set_random_seed,
    train_model,
)


# Set baseline options
HIDDEN_SIZES = [0, 4, 8, 16, 32]
BASE_SEED = 42

# Set project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]


def select_better_model(candidate, current_best):
    # Select by validation performance
    if current_best is None:
        return True

    candidate_f1 = candidate["validation_macro_f1"]
    best_f1 = current_best["validation_macro_f1"]

    if candidate_f1 != best_f1:
        return candidate_f1 > best_f1

    if candidate["validation_loss"] != current_best["validation_loss"]:
        return candidate["validation_loss"] < current_best["validation_loss"]

    return candidate["hidden_size"] < current_best["hidden_size"]


def run_baseline(dataset_names=None, folds=None, output_name="baseline"):
    # Set requested datasets and folds
    datasets = load_clean_datasets()
    if dataset_names is None:
        dataset_names = list(datasets.keys())
    if folds is None:
        folds = list(range(1, 6))

    results_dir = PROJECT_DIR / "results" / output_name
    figures_dir = PROJECT_DIR / "figures" / output_name
    models_dir = PROJECT_DIR / "models" / output_name
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = []
    all_candidate_results = []

    for dataset_name in dataset_names:
        # Prepare dataset folds
        X = datasets[dataset_name]["X"]
        y = datasets[dataset_name]["y"]
        fold_assignments = create_fold_assignments(X, y)
        dataset_predictions = []

        for fold in folds:
            # Prepare scaled fold data
            run = prepare_run(X, y, fold_assignments, fold)
            input_size = run["X_train"].shape[1]
            class_labels = run["label_encoder"].classes_
            output_size = len(class_labels)
            best_candidate = None

            print(f"\nBaseline: {dataset_name}, fold {fold}")

            for hidden_size in HIDDEN_SIZES:
                # Create candidate model
                seed = BASE_SEED + fold
                set_random_seed(seed)
                model = ExpandableNeuralNetwork(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_size=hidden_size,
                )

                # Train candidate model
                training_result = train_model(
                    model=model,
                    X_train=run["X_train"],
                    y_train=run["y_train"],
                    X_validation=run["X_validation"],
                    y_validation=run["y_validation"],
                    seed=seed,
                )

                candidate = {
                    "dataset": dataset_name,
                    "fold": fold,
                    "hidden_size": hidden_size,
                    "epochs": training_result["epochs"],
                    "training_macro_f1": training_result["training_metrics"]["macro_f1"],
                    "validation_macro_f1": training_result["validation_metrics"]["macro_f1"],
                    "validation_loss": training_result["validation_loss"],
                    "model": training_result["model"],
                    "history": training_result["history"],
                }

                all_candidate_results.append(
                    {key: value for key, value in candidate.items() if key not in ["model", "history"]}
                )

                # Save candidate history
                history_path = (
                    results_dir
                    / "training_history"
                    / f"{dataset_name}_fold_{fold}_hidden_{hidden_size}.csv"
                )
                history_path.parent.mkdir(parents=True, exist_ok=True)
                candidate["history"].to_csv(history_path, index=False)

                print(
                    f"  hidden={hidden_size:2d}, "
                    f"validation macro-F1={candidate['validation_macro_f1']:.4f}, "
                    f"validation loss={candidate['validation_loss']:.4f}"
                )

                if select_better_model(candidate, best_candidate):
                    best_candidate = candidate

            # Evaluate selected model
            test_metrics, test_predictions = evaluate_model(
                best_candidate["model"],
                run["X_test"],
                run["y_test"],
            )

            fold_result = {
                "dataset": dataset_name,
                "fold": fold,
                "hidden_size": best_candidate["hidden_size"],
                "epochs": best_candidate["epochs"],
                **test_metrics,
            }
            all_fold_metrics.append(fold_result)

            # Save fold predictions
            prediction_table = pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "fold": fold,
                    "observation_id": run["test_indices"],
                    "true_encoded": run["y_test"],
                    "predicted_encoded": test_predictions,
                    "true_label": class_labels[run["y_test"]],
                    "predicted_label": class_labels[test_predictions],
                }
            )
            dataset_predictions.append(prediction_table)

            # Save selected model
            model_path = models_dir / f"{dataset_name}_fold_{fold}.pt"
            torch.save(
                {
                    "model_config": best_candidate["model"].get_config(),
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in best_candidate["model"].state_dict().items()
                    },
                    "class_labels": class_labels.tolist(),
                    "hidden_size": best_candidate["hidden_size"],
                    "fold": fold,
                },
                model_path,
            )

            # Save fold evaluation outputs
            save_classification_outputs(
                y_true=run["y_test"],
                y_pred=test_predictions,
                class_labels=class_labels,
                results_path=(
                    results_dir
                    / "classification_reports"
                    / f"{dataset_name}_fold_{fold}.csv"
                ),
                figure_path=(
                    figures_dir
                    / "confusion_matrices"
                    / f"{dataset_name}_fold_{fold}.png"
                ),
                title=f"Baseline {dataset_name} fold {fold}",
            )

            print(
                f"  selected hidden={best_candidate['hidden_size']}, "
                f"test macro-F1={test_metrics['macro_f1']:.4f}"
            )

        # Save dataset predictions
        predictions = pd.concat(dataset_predictions, ignore_index=True)
        predictions.to_csv(
            results_dir / f"{dataset_name}_test_predictions.csv",
            index=False,
        )

        # Save combined out-of-fold outputs
        if sorted(folds) == list(range(1, 6)):
            predictions = predictions.sort_values("observation_id")
            save_classification_outputs(
                y_true=predictions["true_encoded"].to_numpy(),
                y_pred=predictions["predicted_encoded"].to_numpy(),
                class_labels=class_labels,
                results_path=(
                    results_dir
                    / "classification_reports"
                    / f"{dataset_name}_combined.csv"
                ),
                figure_path=(
                    figures_dir
                    / "confusion_matrices"
                    / f"{dataset_name}_combined.png"
                ),
                title=f"Baseline {dataset_name} combined test predictions",
            )

    # Save experiment summaries
    fold_metrics = pd.DataFrame(all_fold_metrics)
    candidate_results = pd.DataFrame(all_candidate_results)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    candidate_results.to_csv(results_dir / "candidate_validation_results.csv", index=False)

    return fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", "dry_bean", "yeast", "wine_quality"],
        default="all",
    )
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    dataset_names = None if args.dataset == "all" else [args.dataset]
    folds = None if args.fold is None else [args.fold]
    output_name = "pilot/baseline" if args.pilot else "baseline"

    if args.pilot:
        dataset_names = ["dry_bean"]
        folds = [1]

    run_baseline(dataset_names, folds, output_name)


if __name__ == "__main__":
    main()
