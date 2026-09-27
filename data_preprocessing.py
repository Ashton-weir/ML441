from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# Set project folders
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "preprocessing"
FIGURES_DIR = PROJECT_DIR / "figures" / "preprocessing"


def create_output_folders():
    # Create output folders
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dry_bean():
    # Load Dry Bean data
    file_path = DATA_DIR / "Dry_Bean_Dataset.xlsx"
    data = pd.read_excel(file_path)

    # Separate features and target
    X = data.drop(columns="Class")
    y = data["Class"].astype(str)

    return X, y


def load_yeast():
    # Set Yeast column names
    columns = [
        "sequence_name",
        "mcg",
        "gvh",
        "alm",
        "mit",
        "erl",
        "pox",
        "vac",
        "nuc",
        "localization_site",
    ]

    # Load Yeast data
    file_path = DATA_DIR / "yeast.data"
    data = pd.read_csv(file_path, sep=r"\s+", header=None, names=columns)

    # Remove identifier column
    data = data.drop(columns="sequence_name")

    # Separate features and target
    X = data.drop(columns="localization_site")
    y = data["localization_site"].astype(str)

    return X, y


def load_wine_quality():
    # Load red and white wine data
    red_path = DATA_DIR / "winequality-red.csv"
    white_path = DATA_DIR / "winequality-white.csv"
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")

    # Add wine type variable
    red["wine_type"] = 0
    white["wine_type"] = 1

    # Combine datasets
    data = pd.concat([red, white], ignore_index=True)

    # Separate features and target
    X = data.drop(columns="quality")
    y = data["quality"].astype(str)

    return X, y


def check_dataset(name, X, y):
    # Check missing values
    if X.isna().any().any() or y.isna().any():
        raise ValueError(f"{name}: missing values were found.")


def remove_exact_duplicates(X, y):
    # Combine features and target
    complete_data = X.copy()
    complete_data["target"] = y.values

    # Remove exact duplicate rows
    original_observations = len(complete_data)
    complete_data = complete_data.drop_duplicates().reset_index(drop=True)
    removed_duplicates = original_observations - len(complete_data)

    # Separate features and target
    X_clean = complete_data.drop(columns="target")
    y_clean = complete_data["target"]

    return X_clean, y_clean, original_observations, removed_duplicates


def load_clean_datasets():
    # Load all datasets
    raw_datasets = {
        "dry_bean": load_dry_bean(),
        "yeast": load_yeast(),
        "wine_quality": load_wine_quality(),
    }

    cleaned_datasets = {}

    for name, (X, y) in raw_datasets.items():
        # Check dataset
        check_dataset(name, X, y)

        # Remove exact duplicates
        X, y, original_observations, removed_duplicates = remove_exact_duplicates(X, y)

        cleaned_datasets[name] = {
            "X": X,
            "y": y,
            "original_observations": original_observations,
            "removed_duplicates": removed_duplicates,
        }

    return cleaned_datasets


def save_dataset_summary(
    name,
    X,
    y,
    original_observations,
    removed_duplicates,
):
    # Calculate dataset summary
    summary = {
        "dataset": name,
        "original_observations": original_observations,
        "observations_after_duplicates": len(X),
        "features": X.shape[1],
        "classes": y.nunique(),
        "missing_feature_values": int(X.isna().sum().sum()),
        "missing_target_values": int(y.isna().sum()),
        "removed_duplicates": removed_duplicates,
    }

    # Calculate class distribution
    class_distribution = y.value_counts().rename_axis("class_label").reset_index(name="count")
    class_distribution["percentage"] = (
        100 * class_distribution["count"] / len(y)
    ).round(2)
    class_distribution = class_distribution.sort_values(
        ["count", "class_label"], ascending=[True, True]
    ).reset_index(drop=True)
    class_distribution["incremental_order"] = range(1, len(class_distribution) + 1)

    # Save tables
    class_distribution.to_csv(
        RESULTS_DIR / f"{name}_class_distribution.csv", index=False
    )
    X.describe().T.to_csv(RESULTS_DIR / f"{name}_feature_summary.csv")

    # Plot class distribution
    plt.figure(figsize=(8, 5))
    plt.bar(class_distribution["class_label"], class_distribution["count"])
    plt.xlabel("Class")
    plt.ylabel("Number of observations")
    plt.title(f"{name.replace('_', ' ').title()} class distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"{name}_class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return summary, class_distribution


def main():
    # Create output folders
    create_output_folders()

    # Load and clean all datasets
    datasets = load_clean_datasets()

    summaries = []

    for name, dataset in datasets.items():
        X = dataset["X"]
        y = dataset["y"]
        original_observations = dataset["original_observations"]
        removed_duplicates = dataset["removed_duplicates"]

        # Summarise cleaned dataset
        summary, class_distribution = save_dataset_summary(
            name,
            X,
            y,
            original_observations,
            removed_duplicates,
        )
        summaries.append(summary)

        print(f"\n{name.replace('_', ' ').title()}")
        print(f"Shape: {X.shape}")
        print(f"Exact duplicates removed: {removed_duplicates}")
        print(class_distribution.to_string(index=False))

    # Save combined dataset summary
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    print("\nDataset summary")
    print(summary_table.to_string(index=False))
    print(f"\nTables saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
