from __future__ import annotations

import importlib
import os

import pandas as pd


def test_target_per_class_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TARGET_PER_CLASS", "17")
    mod = importlib.import_module("scripts.build_all_results")
    mod = importlib.reload(mod)
    assert mod.TARGET_PER_CLASS == 17


def test_expand_feature_table_balances_classes() -> None:
    mod = importlib.import_module("scripts.build_all_results")

    df = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "class_id": [0, 1, 2, 3],
            "class_name": ["Viable", "Early Apoptosis", "Late Apoptosis", "Necrosis"],
            **{col: [1.0, 2.0, 3.0, 4.0] for col in mod.FEATURE_COLS},
        }
    )

    expanded = mod._expand_feature_table(df, target_per_class=3, seed=123)
    counts = expanded["class_id"].value_counts().to_dict()
    assert all(counts[cid] >= 3 for cid in range(4))
    assert set(mod.FEATURE_COLS).issubset(expanded.columns)
