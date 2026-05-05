"""Tests for deep time series classifiers (#152).

ROCKET/MiniROCKET classifiers require no extra deps.
InceptionTime/ResNet require torch.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture()
def train_data():
    """Training data: sine-like vs constant, 4 series each."""
    rng = np.random.default_rng(42)
    n = 30
    rows: list[dict] = []
    for sid, label in [("A", "sine"), ("B", "sine"), ("C", "sine"), ("D", "sine")]:
        t = np.linspace(0, 4 * np.pi, n)
        y = np.sin(t) + rng.normal(0, 0.1, n)
        for i in range(n):
            rows.append({"unique_id": sid, "ds": i, "y": float(y[i]), "label": label})
    for sid, label in [("E", "const"), ("F", "const"), ("G", "const"), ("H", "const")]:
        y = 5.0 + rng.normal(0, 0.1, n)
        for i in range(n):
            rows.append({"unique_id": sid, "ds": i, "y": float(y[i]), "label": label})
    return pl.DataFrame(rows)


@pytest.fixture()
def test_data():
    """Test data: one sine, one constant (no labels)."""
    rng = np.random.default_rng(99)
    n = 30
    rows: list[dict] = []
    t = np.linspace(0, 4 * np.pi, n)
    for i in range(n):
        rows.append({"unique_id": "X", "ds": i, "y": float(np.sin(t[i]) + rng.normal(0, 0.1))})
    for i in range(n):
        rows.append({"unique_id": "Y", "ds": i, "y": float(5.0 + rng.normal(0, 0.1))})
    return pl.DataFrame(rows)


# ── RocketClassifier tests ───────────────────────────────────────────────


class TestRocketClassifier:
    def test_fit_predict(self, train_data, test_data):
        pytest.importorskip("sklearn")
        from polars_ts.classification.rocket_classifier import RocketClassifier

        clf = RocketClassifier(n_kernels=100)
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        assert "unique_id" in result.columns
        assert "predicted_label" in result.columns
        assert result.shape[0] == 2

    def test_accuracy(self, train_data, test_data):
        pytest.importorskip("sklearn")
        from polars_ts.classification.rocket_classifier import RocketClassifier

        clf = RocketClassifier(n_kernels=200)
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list(), strict=False))
        assert preds["X"] == "sine"
        assert preds["Y"] == "const"

    def test_predict_before_fit_raises(self, test_data):
        from polars_ts.classification.rocket_classifier import RocketClassifier

        clf = RocketClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(test_data)

    def test_self_classification(self, train_data):
        pytest.importorskip("sklearn")
        from polars_ts.classification.rocket_classifier import RocketClassifier

        clf = RocketClassifier(n_kernels=200)
        clf.fit(train_data, label_col="label")
        test = train_data.select("unique_id", "ds", "y")
        result = clf.predict(test)
        # Should get most labels right on training data
        merged = result.join(
            train_data.group_by("unique_id").agg(pl.col("label").first()),
            on="unique_id",
        )
        accuracy = (merged["predicted_label"] == merged["label"]).mean()
        assert accuracy >= 0.75


# ── MiniRocketClassifier tests ───────────────────────────────────────────


class TestMiniRocketClassifier:
    def test_fit_predict(self, train_data, test_data):
        pytest.importorskip("sklearn")
        from polars_ts.classification.rocket_classifier import MiniRocketClassifier

        clf = MiniRocketClassifier(n_kernels=100)
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        assert "unique_id" in result.columns
        assert "predicted_label" in result.columns
        assert result.shape[0] == 2

    def test_accuracy(self, train_data, test_data):
        pytest.importorskip("sklearn")
        from polars_ts.classification.rocket_classifier import MiniRocketClassifier

        clf = MiniRocketClassifier(n_kernels=200)
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list(), strict=False))
        assert preds["X"] == "sine"
        assert preds["Y"] == "const"


# ── InceptionTimeClassifier tests (requires torch) ──────────────────────


class TestInceptionTimeClassifier:
    def test_fit_predict(self, train_data, test_data):
        pytest.importorskip("torch")
        from polars_ts.classification.inception_time import InceptionTimeClassifier

        clf = InceptionTimeClassifier(max_epochs=3)
        clf.fit(train_data, label_col="label")
        assert clf.is_fitted_

        result = clf.predict(test_data)
        assert "unique_id" in result.columns
        assert "predicted_label" in result.columns
        assert result.shape[0] == 2

    def test_accuracy(self, train_data, test_data):
        pytest.importorskip("torch")
        from polars_ts.classification.inception_time import InceptionTimeClassifier

        clf = InceptionTimeClassifier(max_epochs=20, n_filters=16)
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list(), strict=False))
        assert preds["X"] == "sine"
        assert preds["Y"] == "const"

    def test_predict_before_fit_raises(self, test_data):
        pytest.importorskip("torch")
        from polars_ts.classification.inception_time import InceptionTimeClassifier

        clf = InceptionTimeClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(test_data)


# ── ResNetClassifier tests (requires torch) ──────────────────────────────


class TestResNetClassifier:
    def test_fit_predict(self, train_data, test_data):
        pytest.importorskip("torch")
        from polars_ts.classification.resnet_classifier import ResNetClassifier

        clf = ResNetClassifier(max_epochs=3)
        clf.fit(train_data, label_col="label")
        assert clf.is_fitted_

        result = clf.predict(test_data)
        assert "unique_id" in result.columns
        assert "predicted_label" in result.columns
        assert result.shape[0] == 2

    def test_accuracy(self, train_data, test_data):
        pytest.importorskip("torch")
        from polars_ts.classification.resnet_classifier import ResNetClassifier

        clf = ResNetClassifier(max_epochs=20, n_filters=32)
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list(), strict=False))
        assert preds["X"] == "sine"
        assert preds["Y"] == "const"

    def test_predict_before_fit_raises(self, test_data):
        pytest.importorskip("torch")
        from polars_ts.classification.resnet_classifier import ResNetClassifier

        clf = ResNetClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(test_data)


# ── Import tests ─────────────────────────────────────────────────────────


def test_rocket_classifier_importable():
    from polars_ts.classification.rocket_classifier import RocketClassifier

    assert RocketClassifier is not None


def test_minirocket_classifier_importable():
    from polars_ts.classification.rocket_classifier import MiniRocketClassifier

    assert MiniRocketClassifier is not None
