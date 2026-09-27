import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_preprocessing import load_clean_datasets
from data_splitting import create_fold_assignments, prepare_run
from neural_network import ExpandableNeuralNetwork
from training_utils import (
    calculate_metrics,
    evaluate_model,
    save_classification_outputs,
    set_random_seed,
    train_model,
)


# Set incremental rules
BASE_SEED = 42
ACCEPTABLE_MACRO_F1 = 0.90
MAX_HIDDEN_NEURONS = 32

# Set project folder
PROJECT_DIR = Path(__file__).resolve().parents[1]


def select_active_data(X, y, active_classes):
    # Select currently active classes
    mask = np.isin(y, active_classes)
    selected_X = X[mask]
    selected_y_global = y[mask]
    class_to_local = {
        class_label: local_label
        for local_label, class_label in enumerate(active_classes)
    }
    selected_y_local = np.array(
        [class_to_local[class_label] for class_label in selected_y_global],
        dtype=np.int64,
    )

    return selected_X, selected_y_local


def decide_stage_action(training_macro_f1, validation_macro_f1, hidden_size):
    # Check underfitting
    if training_macro_f1 < ACCEPTABLE_MACRO_F1:
        if hidden_size < MAX_HIDDEN_NEURONS:
            return "add_hidden_neuron"
        return "hidden_limit"

    # Check overfitting
    if validation_macro_f1 < ACCEPTABLE_MACRO_F1:
        return "overfitting"

    return "acceptable"


def run_incremental(dataset_names=None, folds=None, output_name="incremental"):
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
    all_stage_results = []

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
            ordered_class_names = run["class_order"]["class_label"].tolist()
            ordered_classes = run["label_encoder"].transform(ordered_class_names).tolist()
            active_classes = ordered_classes[:2]
            seed = BASE_SEED + fold

            # Create zero-hidden-neuron model
            set_random_seed(seed)
            model = ExpandableNeuralNetwork(
                input_size=input_size,
                output_size=2,
                hidden_size=0,
            )

            fold_histories = []
            stage_number = 1
            print(f"\nIncremental: {dataset_name}, fold {fold}")

            while True:
                attempt = 1

                while True:
                    # Select active training and validation data
                    X_train_stage, y_train_stage = select_active_data(
                        run["X_train"],
                        run["y_train"],
                        active_classes,
                    )
                    X_validation_stage, y_validation_stage = select_active_data(
                        run["X_validation"],
                        run["y_validation"],
                        active_classes,
                    )

                    # Train current architecture
                    training_seed = seed + stage_number * 100 + attempt
                    training_result = train_model(
                        model=model,
                        X_train=X_train_stage,
                        y_train=y_train_stage,
                        X_validation=X_validation_stage,
                        y_validation=y_validation_stage,
                        seed=training_seed,
                    )
                    model = training_result["model"]
                    training_macro_f1 = training_result["training_metrics"]["macro_f1"]
                    validation_macro_f1 = training_result["validation_metrics"]["macro_f1"]
                    action = decide_stage_action(
                        training_macro_f1,
                        validation_macro_f1,
                        model.hidden_size,
                    )

                    # Save stage result
                    stage_result = {
                        "dataset": dataset_name,
                        "fold": fold,
                        "stage": stage_number,
                        "attempt": attempt,
                        "active_classes": ",".join(
                            str(class_labels[class_value])
                            for class_value in active_classes
                        ),
                        "number_of_classes": len(active_classes),
                        "hidden_size": model.hidden_size,
                        "training_observations": len(y_train_stage),
                        "validation_observations": len(y_validation_stage),
                        "epochs": training_result["epochs"],
                        "training_macro_f1": training_macro_f1,
                        "validation_macro_f1": validation_macro_f1,
                        "validation_loss": training_result["validation_loss"],
                        "action": action,
                    }
                    all_stage_results.append(stage_result)

                    # Save stage history
                    history = training_result["history"].copy()
                    history.insert(0, "hidden_size", model.hidden_size)
                    history.insert(0, "attempt", attempt)
                    history.insert(0, "stage", stage_number)
                    fold_histories.append(history)

                    print(
                        f"  stage={stage_number}, classes={len(active_classes)}, "
                        f"hidden={model.hidden_size}, "
                        f"train macro-F1={training_macro_f1:.4f}, "
                        f"validation macro-F1={validation_macro_f1:.4f}, "
                        f"action={action}"
                    )

                    if action == "add_hidden_neuron":
                        # Expand hidden layer
                        set_random_seed(training_seed + 1)
                        model.add_hidden_neuron()
                        attempt += 1
                        continue

                    break

                if len(active_classes) == len(ordered_classes):
                    # Finish after all classes are trained
                    break

                # Add next least represented class
                next_class = ordered_classes[len(active_classes)]
                active_classes.append(next_class)
                set_random_seed(seed + stage_number * 1000)
                model.add_output_neuron()
                stage_number += 1

            # Convert final predictions to global labels
            _, local_predictions = evaluate_model(
                model,
                run["X_test"],
                np.array(
                    [active_classes.index(class_value) for class_value in run["y_test"]],
                    dtype=np.int64,
                ),
            )
            test_predictions = np.array(
                [active_classes[local_value] for local_value in local_predictions],
                dtype=np.int64,
            )

            # Calculate final test metrics
            test_metrics = calculate_metrics(run["y_test"], test_predictions)
            fold_result = {
                "dataset": dataset_name,
                "fold": fold,
                "hidden_size": model.hidden_size,
                "stages": stage_number,
                **test_metrics,
            }
            all_fold_metrics.append(fold_result)

            # Save training history
            history_table = pd.concat(fold_histories, ignore_index=True)
            history_path = (
                results_dir
                / "training_history"
                / f"{dataset_name}_fold_{fold}.csv"
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_table.to_csv(history_path, index=False)

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

            # Save final fold model
            model_path = models_dir / f"{dataset_name}_fold_{fold}.pt"
            torch.save(
                {
                    "model_config": model.get_config(),
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "class_labels": class_labels.tolist(),
                    "active_class_order": active_classes,
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
                title=f"Incremental {dataset_name} fold {fold}",
            )

            print(
                f"  final hidden={model.hidden_size}, "
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
                title=f"Incremental {dataset_name} combined test predictions",
            )

    # Save experiment summaries
    fold_metrics = pd.DataFrame(all_fold_metrics)
    stage_results = pd.DataFrame(all_stage_results)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    stage_results.to_csv(results_dir / "stage_results.csv", index=False)

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
    output_name = "pilot/incremental" if args.pilot else "incremental"

    if args.pilot:
        dataset_names = ["dry_bean"]
        folds = [1]

    run_incremental(dataset_names, folds, output_name)


if __name__ == "__main__":
    main()
