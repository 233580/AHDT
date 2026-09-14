"""Bank Marketing example.

Run from the AHDT directory, for example:
    python examples/bank_marketing.py --data ./数据集/bank/bank.csv
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running this file directly from the repository root/AHDT directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_bank_marketing_data  # noqa: E402
from evaluation import cross_validate_ahdt, print_classification_report  # noqa: E402
from model import AHDT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to bank.csv or bank-full.csv")
    args = parser.parse_args()
    X, y, feature_names, _ = load_bank_marketing_data(args.data)

    def model_factory() -> AHDT:
        return AHDT(
            alpha_strategy="auto",
            k_range=(2, 6),
            quantile_threshold=0.6,
            max_depth=8,
            min_samples_split=20,
            random_state=42,
        )

    results, mean_scores, std_scores, details = cross_validate_ahdt(
        X, y, model_factory, n_splits=5, random_state=42
    )
    print(results.to_string(index=False))
    print("\nMean ± std:")
    for metric in mean_scores.index:
        print(f"{metric}: {mean_scores[metric]:.4f} ± {std_scores[metric]:.4f}")
    print("\nSelected features in the last fold:")
    last_model = details[-1]["Model"]
    for idx in last_model.selected_features_:
        print(f"{idx}: {feature_names[idx]}")
    print("\nClassification report (last fold):")
    print_classification_report(details, target_names=["No Subscription", "Subscription"])


if __name__ == "__main__":
    main()
