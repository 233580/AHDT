"""Command-line experiment runner.

Example:
    python run_experiment.py --data ./数据集/bank/bank.csv --delimiter ';' --target y
"""

from __future__ import annotations

import argparse
import os

if __package__:
    from .data import load_tabular_classification_data
    from .evaluation import cross_validate_ahdt, print_classification_report
    from .model import AHDT
else:
    from data import load_tabular_classification_data
    from evaluation import cross_validate_ahdt, print_classification_report
    from model import AHDT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AHDT classification experiment")
    parser.add_argument("--data", required=True, help="Path to a CSV/TSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--delimiter", default=",", help="File delimiter, default: ','")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=20)
    parser.add_argument("--quantile-threshold", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    X, y, feature_names, _ = load_tabular_classification_data(
        path=args.data,
        target_column=args.target,
        delimiter=args.delimiter,
    )
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {len(set(y.tolist()))}")

    def model_factory() -> AHDT:
        return AHDT(
            alpha_strategy="auto",
            k_range=(2, 6),
            quantile_threshold=args.quantile_threshold,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
        )

    results, mean_scores, std_scores, fold_details = cross_validate_ahdt(
        X, y, model_factory=model_factory, n_splits=args.folds, random_state=args.random_state
    )
    print("\nFold results:")
    print(results.to_string(index=False))
    print("\nMean ± std:")
    for metric in mean_scores.index:
        print(f"{metric}: {mean_scores[metric]:.4f} ± {std_scores[metric]:.4f}")

    last_model = fold_details[-1]["Model"]
    print("\nSelected features (last fold):")
    for feature_idx in last_model.selected_features_:
        print(f"  {feature_idx}: {feature_names[feature_idx]}")

    print("\nClassification report (last fold):")
    print_classification_report(fold_details)


if __name__ == "__main__":
    main()
