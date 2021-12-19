"""
The refacted SPOT shall have the same output as the original ones
"""
import os

import numpy as np
import pandas as pd

import pytest

import ads_evt as spot
from ads_evt.spot import moving_average

from .spot_origin import spot as spot_origin


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SPOT_ORIGIN_DIR = os.path.join(BASE_DIR, "spot_origin")


def _read_dat(filename: str) -> np.ndarray:
    with open(filename, encoding="UTF-8") as obj:
        return np.array(list(map(float, obj.read().split(","))))


class Dataset:
    """
    Uniform interface for dataset
    """

    def __init__(
        self, init_data: np.ndarray, data: np.ndarray, proba: float, depth: int
    ):
        self.init_data = init_data
        self.data = data
        self.proba = proba
        self.depth = depth

    @staticmethod
    def prepare_physics() -> "Dataset":
        """
        Prepare dataset from physics.dat
        """
        data = _read_dat(os.path.join(SPOT_ORIGIN_DIR, "physics.dat"))
        return Dataset(init_data=2000, data=data, proba=1e-3, depth=450)

    @staticmethod
    def prepare_rain() -> "Dataset":
        """
        Prepare dataset from rain.dat
        """
        data = _read_dat(os.path.join(SPOT_ORIGIN_DIR, "rain.dat"))
        return Dataset(init_data=1000, data=data, proba=1e-4, depth=200)

    @staticmethod
    def prepare_mawi() -> "Dataset":
        """
        Prepare dataset from MAWI
        """
        data_17 = pd.read_csv(os.path.join(SPOT_ORIGIN_DIR, "mawi_170812_50_50.csv"))
        data_18 = pd.read_csv(os.path.join(SPOT_ORIGIN_DIR, "mawi_180812_50_50.csv"))
        n_init = 1000
        return Dataset(
            init_data=data_17["rSYN"].values[-n_init:],
            data=data_18["rSYN"].values,
            proba=1e-4,
            depth=200,
        )

    @staticmethod
    def prepare_edf_stocks() -> "Dataset":
        """
        Prepare dataset from MAWI
        """
        edf_stocks = pd.read_csv(os.path.join(SPOT_ORIGIN_DIR, "edf_stocks.csv"))

        # stream
        u_data = edf_stocks["DATE"] == "2017-02-09"
        data: np.ndarray = edf_stocks["LOW"][u_data].values

        # initial batch
        u_init_data = (
            (edf_stocks["DATE"] == "2017-02-08")
            | (edf_stocks["DATE"] == "2017-02-07")
            | (edf_stocks["DATE"] == "2017-02-06")
        )
        init_data: np.ndarray = edf_stocks["LOW"][u_init_data].values

        return Dataset(init_data=init_data, data=data, proba=1e-5, depth=10)


@pytest.mark.parametrize(
    "dataset",
    [
        Dataset.prepare_physics(),
        Dataset.prepare_rain(),
        Dataset.prepare_mawi(),
        Dataset.prepare_edf_stocks(),
    ],
)
def test_moving_average(dataset: Dataset):
    """
    ads_evt.spot.moving_average shall behave the same as backMean
    """
    np.testing.assert_array_equal(
        moving_average(dataset.data, dataset.depth),
        spot_origin.backMean(dataset.data, dataset.depth),
    )


def _compare(model: spot.SPOTBase, model_origin: spot.SPOTBase, dataset: Dataset):
    results = []
    for alg in [model, model_origin]:
        alg.fit(init_data=dataset.init_data, data=dataset.data)
        alg.initialize()
        results.append(alg.run())

    for key in results[0]:
        assert results[0][key] == pytest.approx(results[1][key], nan_ok=True)


@pytest.mark.parametrize(
    "dataset",
    [
        Dataset.prepare_physics(),
        Dataset.prepare_mawi(),
        Dataset.prepare_edf_stocks(),
    ],
)
def test_spot(dataset: Dataset):
    """
    End-to-end comparison
    """
    _compare(spot.SPOT(q=dataset.proba), spot_origin.SPOT(q=dataset.proba), dataset)
    _compare(
        spot.dSPOT(q=dataset.proba, depth=dataset.depth),
        spot_origin.dSPOT(q=dataset.proba, depth=dataset.depth),
        dataset,
    )
    _compare(spot.biSPOT(q=dataset.proba), spot_origin.biSPOT(q=dataset.proba), dataset)
    # The original implementation of bidSPOT uses n_points=8 for _grimshaw by default
    _compare(
        spot.bidSPOT(q=dataset.proba, depth=dataset.depth, n_points=8),
        spot_origin.bidSPOT(q=dataset.proba, depth=dataset.depth),
        dataset,
    )


def test_no_peaks():
    """
    biSPOT shall handle no initial peaks, while the original one will raise error
    """
    dataset = Dataset.prepare_rain()
    with pytest.raises(ValueError):
        alg = spot_origin.biSPOT(q=dataset.proba)
        alg.fit(init_data=dataset.init_data, data=dataset.data)
        alg.initialize()
    alg = spot.biSPOT(q=dataset.proba)
    alg.fit(init_data=dataset.init_data, data=dataset.data)
    alg.initialize()
    _ = alg.run()
