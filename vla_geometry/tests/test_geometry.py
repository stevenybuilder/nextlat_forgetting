import numpy as np
import pytest

from vla_geometry.geometry import (
    AdditiveFactorModel,
    assign_combination_folds,
    crossfit_interference,
)
from vla_geometry.grid import build_cells


FACTORS = {
    "target_shape": ["a", "b", "c"],
    "target_texture": ["red", "blue"],
    "receptacle_shape": ["bowl", "square"],
    "receptacle_texture": ["plain", "stripe"],
}


def _additive_fixture():
    cells = build_cells(FACTORS)
    rng = np.random.default_rng(12)
    effects = {
        factor: {value: rng.normal(size=16) for value in values}
        for factor, values in FACTORS.items()
    }
    embeddings = np.stack(
        [
            sum((effects[factor][cell[factor]] for factor in FACTORS), start=np.zeros(16))
            for cell in cells
        ]
    )
    labels = {
        factor: np.asarray([cell[factor] for cell in cells]) for factor in FACTORS
    }
    return cells, embeddings, labels


def test_additive_model_reconstructs_exact_factor_sum():
    _, embeddings, labels = _additive_fixture()
    model = AdditiveFactorModel.fit(embeddings, labels)
    assert model.reconstruction_r2(embeddings, labels) > 0.999999


def test_crossfit_excludes_whole_cells_and_reconstructs():
    cells, embeddings, _ = _additive_fixture()
    result = crossfit_interference(
        embeddings, cells, tuple(FACTORS), n_folds=5, seed=9
    )
    assert result["overall_r2"] > 0.999999
    assert np.all(np.isfinite(result["interference"]))
    assert len(result["folds"]) == len(cells)


def test_fold_assignment_rejects_missing_level_coverage():
    cells = [
        {"a": "rare", "b": str(idx), "cell_id": str(idx)}
        if idx == 0
        else {"a": "common", "b": str(idx), "cell_id": str(idx)}
        for idx in range(6)
    ]
    with pytest.raises(ValueError, match="drops levels"):
        assign_combination_folds(cells, ("a", "b"), n_folds=2, seed=2)


def test_unknown_level_is_not_silently_encoded():
    _, embeddings, labels = _additive_fixture()
    model = AdditiveFactorModel.fit(embeddings, labels)
    altered = {factor: values[:1].copy() for factor, values in labels.items()}
    altered["target_shape"][0] = "never-seen"
    with pytest.raises(ValueError, match="unseen levels"):
        model.predict(altered)

