"""
Test suites for spot
"""
import csv
import os

import numpy as np

from ads_evt import biSPOT
from ads_evt.spot import ExtremeValue


_BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def test_zero_peaks():
    """
    ExtremeValue shall be robust to only one peak and peak of zero
    """
    init_threshold = 1.0
    evt = ExtremeValue(q=0.01, key=lambda item: -item)
    evt.initialize(np.array([]), init_threshold=init_threshold)
    assert evt.num_peaks == 0

    # With the only one peak, peaks.min() == peaks.mean(),
    # which leads to b == c for root finder
    assert evt.run(0, num=100, with_alarm=False) == evt.Status.ABNORMAL

    # With extreme_quantile "lower" than init_threshold,
    # a new data point of init_threshold will introduce a peak of zero.
    # Hence, peaks.min() == 0, which leads to zero-division.
    # With a desc key, b is -infty, while c is +infty.
    #
    # "Lower" here means a number may exceed extreme_quantile but not init_threshold.
    assert evt.extreme_quantile > init_threshold
    assert evt.run(init_threshold, num=101, with_alarm=False) == evt.Status.ABNORMAL


def test_large_peak():
    """
    The default epsilon for _grimshaw shall be updated for large peaks
    """
    evt = ExtremeValue()
    # The original implementation updates epsilon (1e-8) in _grimshaw if
    # abs(a) < 2 * epsilon
    # However, roots will be searched in [((a + epsilon) + epsilon), -epsilon]
    # Hence, the maximum peak between 0.33e8 and 0.5e8 will lead to failure
    evt.initialize(np.array([0.4e8]), init_threshold=1.0)


def test_close_mean_and_min():
    """
    When y_mean and y_min are almost the same due to floating point errors,
    ExtremeValue should not raise errors
    """
    with open(os.path.join(_BASE_DIR, "sample.csv"), encoding="UTF-8") as obj:
        data = np.array(list(csv.reader(obj)), dtype=float)
    model = biSPOT()
    model.fit(init_data=data[:-5, 0], data=data[-5:, 0])
    model.initialize()
    _ = model.run(with_alarm=False)
