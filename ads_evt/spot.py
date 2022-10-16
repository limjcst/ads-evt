"""
SPOT algorithms for anomaly detection edited from the following repository,
which is released under the GNU GPLv3 license
https://github.com/Amossys-team/SPOT
"""
# pylint: disable=invalid-name
from abc import ABC
from enum import Enum
from enum import auto
import json
import logging
from typing import Callable
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


_Template = TypeVar("_Template")


def _asc_key(value: _Template) -> _Template:
    return value


def _desc_key(value: _Template) -> _Template:
    return -value


class ExtremeValue:
    """
    Extreme value with one threshold
    """

    class Status(Enum):
        """
        Detection result
        """

        NORMAL = auto()
        ABNORMAL = auto()
        ALARM = auto()

    def __init__(
        self,
        q: float = 1e-4,
        n_points: int = 10,
        key: Callable[[_Template], _Template] = _asc_key,
        logging_level: int = logging.WARNING,
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
        """
        self._proba = q
        self._n_points = n_points
        self._key = key

        self._extreme_quantile: float = None
        self._init_threshold: float = None
        self._peaks: np.ndarray = None

        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._logger.setLevel(level=logging_level)

    @property
    def extreme_quantile(self) -> float:
        """
        current threshold (bound between normal and abnormal events)
        """
        return self._extreme_quantile

    @property
    def num_peaks(self) -> int:
        """
        number of observed peaks
        """
        return self._peaks.size

    def summary(self) -> dict:
        """
        Summary running status
        """
        return {
            "Detection level q": self._proba,
            "initial threshold": self._init_threshold,
            "#(peaks)": self.num_peaks,
            "extreme quantile": self._extreme_quantile,
        }

    @staticmethod
    def _roots_finder(
        fun: Callable[[float], float],
        jac: Callable[[float], float],
        bounds: Tuple[float, float],
        npoints: int,
        method: str,
    ) -> np.ndarray:
        """
        Find possible roots of a scalar function

        Parameters:
            fun: scalar function
            jac: first order derivative of the function
            bounds: (min,max) interval for the roots search
            npoints: maximum number of roots to output
            method:
                'regular' : regular sample of the search interval,
                'random' : uniform (distribution) sample of the search interval

        Returns: possible roots of the function
        """
        if method == "regular":
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            initial_guess = np.arange(bounds[0] + step, bounds[1], step)
        elif method == "random":
            initial_guess = np.random.uniform(bounds[0], bounds[1], npoints)

        def _object(variable: np.ndarray) -> Tuple[float, np.ndarray]:
            value = np.array([fun(item) for item in variable])
            gradient = np.array([jac(item) for item in variable])
            return (value**2).sum(), 2 * value * gradient

        opt = minimize(
            _object,
            initial_guess,
            method="L-BFGS-B",
            jac=True,
            bounds=[bounds] * len(initial_guess),
        )

        X: np.ndarray = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    @staticmethod
    def _log_likelihood(Y: np.ndarray, gamma: float, sigma: float) -> float:
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters:
            Y: observations
            gamma: GPD index parameter
            sigma: GPD scale parameter (>0)

        Returns: log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * np.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + np.log(Y.mean()))
        return L

    def _grimshaw(
        self, peaks: np.ndarray, epsilon: float = 1e-8
    ) -> Tuple[float, float, float]:
        # pylint: disable=too-many-locals
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters:
            epsilon: numerical parameter to perform (default : 1e-8)

        Returns: gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def _u(s: np.ndarray) -> float:
            return 1 + np.log(s).mean()

        def _v(s: np.ndarray) -> float:
            return np.mean(1 / s)

        def _w(t: float) -> float:
            s = 1 + t * peaks
            us = _u(s)
            vs = _v(s)
            return us * vs - 1

        def _jac_w(t: float) -> float:
            s = 1 + t * peaks
            us = _u(s)
            vs = _v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s**2))
            return us * jac_vs + vs * jac_us

        y_min: float = peaks.min()
        y_max: float = peaks.max()
        y_mean: float = peaks.mean()

        a = -1 / y_max
        if abs(a) < 3 * epsilon:
            epsilon = abs(a) / self._n_points

        a = a + epsilon

        # We look for possible roots
        left_zeros = self._roots_finder(
            _w,
            _jac_w,
            (a + epsilon, -epsilon),
            self._n_points,
            "regular",
        )

        if y_mean > y_min > 0 and not np.isclose(y_mean, y_min):
            b = 2 * (y_mean - y_min) / (y_mean * y_min)
            c = 2 * (y_mean - y_min) / (y_min**2)
            right_zeros = self._roots_finder(
                _w,
                _jac_w,
                (b, c),
                self._n_points,
                "regular",
            )
            # all the possible roots
            zeros = np.concatenate((left_zeros, right_zeros))
        else:
            zeros = left_zeros

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = y_mean
        ll_best = self._log_likelihood(peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = _u(1 + z * peaks) - 1
            sigma = gamma / z
            ll = self._log_likelihood(peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, num: int, gamma: float, sigma: float) -> float:
        """
        Compute the quantile at level 1-q

        Parameters:
            gamma: GPD parameter
            sigma: GPD parameter

        Returns: quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = num * self._proba / self.num_peaks
        if gamma != 0:
            return self._init_threshold + self._key(
                (sigma / gamma) * (pow(r, -gamma) - 1)
            )
        return self._init_threshold - self._key(sigma * np.log(r))

    def initialize(self, data: np.ndarray, init_threshold: float):
        """
        Run the calibration (initialization) step
        """
        self._init_threshold = init_threshold

        # initial peaks
        self._peaks = self._key(
            data[self._key(data) > self._key(self._init_threshold)]
            - self._init_threshold
        )

        self._logger.debug("Initial threshold : %s", self._init_threshold)
        self._logger.debug("Number of peaks : %s", self.num_peaks)
        self._logger.debug("Grimshaw maximum log-likelihood estimation ... ")

        if self._peaks.size:
            gamma, sigma, ll = self._grimshaw(self._peaks)
            self._extreme_quantile = self._quantile(data.size, gamma, sigma)
            self._logger.debug(
                "gamma = %s, sigma = %s, log-likelihood = %s", gamma, sigma, ll
            )
        else:
            self._extreme_quantile = self._init_threshold
            self._logger.info("Initialized with no peaks")
        self._logger.debug(
            "Extreme quantile (probability = %s): %s",
            self._proba,
            self._extreme_quantile,
        )

    def run(self, datum: float, num: int, with_alarm: bool = True) -> Status:
        """
        Run SPOT on the stream

        Parameters:
            with_alarm: If False, SPOT will adapt the threshold assuming
                there is no abnormal values (default = True)
        """
        # If the observed value exceeds the current threshold (alarm case)
        if self._key(datum) > self._key(self._extreme_quantile):
            # if we want to alarm, we put it in the alarm list
            if with_alarm:
                return self.Status.ALARM
            # otherwise we add it in the peaks
            self._peaks = np.append(
                self._peaks, self._key(datum - self._init_threshold)
            )
            # and we update the thresholds

            g, s, _ = self._grimshaw(self._peaks)
            self._extreme_quantile = self._quantile(num + 1, g, s)

        # case where the value exceeds the initial threshold but not the alarm ones
        elif self._key(datum) > self._key(self._init_threshold):
            # we add it in the peaks
            self._peaks = np.append(
                self._peaks, self._key(datum - self._init_threshold)
            )
            # and we update the thresholds

            g, s, _ = self._grimshaw(self._peaks)
            self._extreme_quantile = self._quantile(num + 1, g, s)
        else:
            return self.Status.NORMAL
        return self.Status.ABNORMAL


class SPOTBase(ABC):
    """
    The base class for the SPOT algorithm with data management
    """

    # colors for plot
    DEEP_SAFFRON = "#FF9933"
    AIR_FORCE_BLUE = "#5D8AA8"
    _plot_keys = ()

    def __init__(self, logging_level: int = logging.WARNING):
        self._data: np.ndarray = None
        self._init_data: np.ndarray = None
        self._num: int = 0

        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._logger.setLevel(level=logging_level)

    def summary(self) -> dict:
        """
        Summar running status
        """
        report = {
            "name": "Streaming Peaks-Over-Threshold Object",
        }
        if self._data is not None:
            report["Data imported"] = "Yes"
            report["#(initialization values)"] = self._init_data.size
            report["#(stream values)"] = self._data.size
        else:
            report["Data imported"] = "No"
            return report

        if self._num == 0:
            report["Algorithm initialized"] = "No"
        else:
            report["Algorithm initialized"] = "Yes"
            rest = self._num - self._init_data.size
            if rest > 0:
                report["Algorithm run"] = "Yes"
                report["#(observations)"] = f"{rest} ({100 * rest / self._num:.2f} %%)"
            else:
                report["Algorithm run"] = "No"
        return report

    def __str__(self):
        return json.dumps(self.summary(), indent=2, ensure_ascii=False)

    def fit(
        self,
        init_data: Union[np.ndarray, pd.Series, list, int, float],
        data: Union[np.ndarray, pd.Series, list],
    ):
        """
        Import data to SPOT object

        Parameters:
            init_data: initial batch to calibrate the algorithm
            data: data for the run
        """
        if isinstance(data, list):
            self._data = np.array(data)
        elif isinstance(data, np.ndarray):
            self._data = data
        elif isinstance(data, pd.Series):
            self._data = data.values
        else:
            self._logger.warning("This data format (%s) is not supported", type(data))
            return

        if isinstance(init_data, list):
            self._init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self._init_data = init_data
        elif isinstance(init_data, pd.Series):
            self._init_data = init_data.values
        elif isinstance(init_data, int):
            self._init_data = self._data[:init_data]
            self._data = self._data[init_data:]
        elif isinstance(init_data, float) and (0 < init_data < 1):
            r = int(init_data * data.size)
            self._init_data = self._data[:r]
            self._data = self._data[r:]
        else:
            self._logger.warning("The initial data cannot be set")
            return

    def add(self, data: Union[np.ndarray, pd.Series, list]):
        """
        This function allows to append data to the already fitted data

        Parameters:
            data: data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values
        elif not isinstance(data, np.ndarray):
            self._logger.warning("This data format (%s) is not supported", type(data))
            return

        self._data = np.append(self._data, data)

    def initialize(self, level: float = 0.98):
        """
        Run the calibration (initialization) step

        Parameters:
            level: Probability associated with the initial threshold t (default 0.98)
        """
        raise NotImplementedError

    def run(self, with_alarm: bool = True) -> dict:
        """
        Run SPOT on the stream

        Parameters:
            with_alarm: If False, SPOT will adapt the threshold assuming
                there is no abnormal values (default = True)
        """
        raise NotImplementedError

    def plot(self, run_results: dict, with_alarm: bool = True):
        """
        Plot the results of given by the run

        Parameters:
            run_results: results given by the 'run' method
            with_alarm: If True, alarms are plotted. (default = True)

        Returns: a list of the plots
        """
        ticks = list(range(self._data.size))

        (ts_fig,) = plt.plot(ticks, self._data, color=self.AIR_FORCE_BLUE)
        fig = [ts_fig]

        for key in self._plot_keys:
            if key in run_results:
                (sub_fig,) = plt.plot(
                    ticks, run_results[key], color=self.DEEP_SAFFRON, lw=2, ls="dashed"
                )
                fig.append(sub_fig)

        if with_alarm and ("alarms" in run_results):
            alarm = run_results["alarms"]
            if alarm:
                fig.append(plt.scatter(alarm, self._data[alarm], color="red"))

        plt.xlim((0, self._data.size))

        return fig


class SPOT(SPOTBase):
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)
    """

    _plot_keys = ("thresholds",)

    def __init__(
        self, q: float = 1e-4, n_points: int = 10, logging_level: int = logging.WARNING
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
        """
        super().__init__(logging_level=logging_level)
        self._ev = ExtremeValue(q=q, n_points=n_points, logging_level=logging_level)

    def summary(self) -> dict:
        report = super().summary()
        report["Extreme Value"] = self._ev.summary()
        return report

    def initialize(self, level: float = 0.98):
        data = self._init_data
        level = level - np.floor(level)

        # t is fixed for the whole algorithm
        init_threshold = sorted(data)[int(level * data.size)]
        self._ev.initialize(data=data, init_threshold=init_threshold)
        self._num = data.size

    def run(self, with_alarm: bool = True) -> dict:
        """
        Run SPOT on the stream

        Parameters:
            with_alarm: If False, SPOT will adapt the threshold assuming
                there is no abnormal values (default = True)

        Returns:
            a dict:
                keys : 'thresholds' and 'alarms'

                'thresholds' contains the extreme quantiles and 'alarms' contains
                the indexes of the values which have triggered alarms

        """
        if self._num > self._init_data.size:
            self._logger.warning(
                "the algorithm seems to have already been run, "
                "you should initialize before running again"
            )
            return {}

        # list of the thresholds
        thresholds = []
        alarms = []
        # Loop over the stream
        for i, datum in enumerate(self._data):
            if (
                self._ev.run(datum, self._num, with_alarm=with_alarm)
                == ExtremeValue.Status.ALARM
            ):
                alarms.append(i)
            else:
                self._num += 1

            thresholds.append(self._ev.extreme_quantile)  # thresholds record

        return dict(thresholds=thresholds, alarms=alarms)


class biSPOT(SPOTBase):
    """
    This class allows to run biSPOT algorithm on univariate dataset (upper and lower bounds)
    """

    _plot_keys = ("upper_thresholds", "lower_thresholds")

    def __init__(
        self, q: float = 1e-4, n_points: int = 10, logging_level: int = logging.WARNING
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
        """
        super().__init__(logging_level=logging_level)
        self._ev = {
            "upper": ExtremeValue(
                q=q, n_points=n_points, key=_asc_key, logging_level=logging_level
            ),
            "lower": ExtremeValue(
                q=q, n_points=n_points, key=_desc_key, logging_level=logging_level
            ),
        }

    def summary(self) -> dict:
        report = super().summary()
        report["upper Extreme Value"] = self._ev["upper"].summary()
        report["lower Extreme Value"] = self._ev["lower"].summary()
        return report

    def initialize(self, level: float = 0.98):
        data = self._init_data
        level = level - np.floor(level)

        _data = sorted(data)
        # t is fixed for the whole algorithm
        init_thresholds = {
            "upper": _data[int(level * data.size)],
            "lower": _data[int((1 - level) * data.size)],
        }
        for key, ev in self._ev.items():
            ev.initialize(data=data, init_threshold=init_thresholds[key])
        self._num = data.size

    def run(self, with_alarm: bool = True) -> dict:
        """
        Run biSPOT on the stream

        Parameters:
            with_alarm: If False, SPOT will adapt the threshold assuming
                there is no abnormal values (default = True)

        Returns:
            a dict:
                keys : 'upper_thresholds', 'lower_thresholds' and 'alarms'

                '*_thresholds' contains the extreme quantiles and 'alarms' contains
                the indexes of the values which have triggered alarms

        """
        if self._num > self._init_data.size:
            self._logger.warning(
                "the algorithm seems to have already been run, "
                "you should initialize before running again"
            )
            return {}

        # list of the thresholds
        thresholds = {key: [] for key in self._ev}
        alarms = []
        # Loop over the stream
        for i, datum in enumerate(self._data):
            ret = {
                ev.run(datum, self._num, with_alarm=with_alarm)
                for ev in self._ev.values()
            }
            if ExtremeValue.Status.ALARM in ret:
                alarms.append(i)
            else:
                self._num += 1
            for key, ev in self._ev.items():
                thresholds[key].append(ev.extreme_quantile)

        return dict(
            upper_thresholds=thresholds["upper"],
            lower_thresholds=thresholds["lower"],
            alarms=alarms,
        )


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Moving average of the given data
    """
    mean: List[float] = []
    accumulation: float = data[:window].sum()
    mean.append(accumulation / window)
    for i in range(window, len(data)):
        accumulation = accumulation - data[i - window] + data[i]
        mean.append(accumulation / window)
    return np.array(mean)


class dSPOT(SPOT):
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper-bound)
    """

    def __init__(
        self,
        q: float = 1e-4,
        n_points: int = 10,
        depth: int = 10,
        logging_level: int = logging.WARNING,
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
            depth: Number of observations to compute the moving average
        """
        super().__init__(q=q, n_points=n_points, logging_level=logging_level)
        self._depth = depth

    def initialize(self, level: float = 0.98):
        data: np.ndarray = (
            self._init_data[self._depth :]
            - moving_average(self._init_data, self._depth)[:-1]
        )
        level = level - np.floor(level)

        # t is fixed for the whole algorithm
        init_threshold = sorted(data)[int(level * data.size)]
        self._ev.initialize(data=data, init_threshold=init_threshold)
        self._num = data.size

    def run(self, with_alarm: bool = True) -> dict:
        if self._num > self._init_data.size:
            self._logger.warning(
                "the algorithm seems to have already been run, "
                "you should initialize before running again"
            )
            return {}

        # actual normal window
        window: np.ndarray = self._init_data[-self._depth :]

        # list of the thresholds
        thresholds = []
        alarms = []
        # Loop over the stream
        for i, datum in enumerate(self._data):
            mean = window.mean()
            if (
                self._ev.run(datum - mean, self._num, with_alarm=with_alarm)
                == ExtremeValue.Status.ALARM
            ):
                alarms.append(i)
            else:
                self._num += 1
                window = np.append(window[1:], datum)

            thresholds.append(self._ev.extreme_quantile + mean)  # thresholds record

        return {"thresholds": thresholds, "alarms": alarms}


class bidSPOT(biSPOT):
    """
    This class allows to run biDSPOT algorithm on univariate dataset (upper and lower bounds)
    """

    def __init__(
        self,
        q: float = 1e-4,
        n_points: int = 10,
        depth: int = 10,
        logging_level: int = logging.WARNING,
    ):
        """
        Constructor

        Parameters:
            q: Detection level (risk)
            n_points: maximum number of candidates for maximum likelihood (default : 10)
            depth: Number of observations to compute the moving average
        """
        super().__init__(q=q, n_points=n_points, logging_level=logging_level)
        self._depth = depth

    def initialize(self, level: float = 0.98):
        data: np.ndarray = (
            self._init_data[self._depth :]
            - moving_average(self._init_data, self._depth)[:-1]
        )
        level = level - np.floor(level)

        _data = sorted(data)
        # t is fixed for the whole algorithm
        init_thresholds = {
            "upper": _data[int(level * data.size)],
            "lower": _data[int((1 - level) * data.size)],
        }
        for key, ev in self._ev.items():
            ev.initialize(data=data, init_threshold=init_thresholds[key])
        self._num = data.size

    def run(self, with_alarm: bool = True):
        if self._num > self._init_data.size:
            self._logger.warning(
                "the algorithm seems to have already been run, "
                "you should initialize before running again"
            )
            return {}

        # actual normal window
        window: np.ndarray = self._init_data[-self._depth :]

        # list of the thresholds
        thresholds = {key: [] for key in self._ev}
        alarms = []
        # Loop over the stream
        for i, datum in enumerate(self._data):
            mean = window.mean()
            ret = {
                ev.run(datum - mean, self._num, with_alarm=with_alarm)
                for ev in self._ev.values()
            }
            if ExtremeValue.Status.ALARM in ret:
                alarms.append(i)
            else:
                self._num += 1
                window = np.append(window[1:], datum)
            for key, ev in self._ev.items():
                thresholds[key].append(ev.extreme_quantile + mean)

        return dict(
            upper_thresholds=thresholds["upper"],
            lower_thresholds=thresholds["lower"],
            alarms=alarms,
        )
