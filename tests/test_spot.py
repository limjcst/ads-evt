"""
Test suites for spot
"""
import numpy as np

from ads_evt.spot import ExtremeValue


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
