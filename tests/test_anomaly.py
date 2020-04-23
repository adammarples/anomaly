import anomaly

import pandas as pd
from unittest import TestCase


class TestAnomalyDetector(TestCase):
    """
    Weak tests that just test the path through each detrend/fit/plot
    option to check for exceptions.

    """
    def setUp(self):
        self.series = pd.read_csv('../data/test.csv', squeeze=True, index_col=0, parse_dates=True)

    def test_fit_default(self):
        detector = anomaly.AnomalyDetector(self.series)
        detector.fit()
        detector.plot()

    def test_fit_default_detrend(self):
        detector = anomaly.AnomalyDetector(self.series)
        detector.detrend(how='linear')
        detector.fit()
        detector.plot()

    def test_fit_default_detrend(self):
        detector = anomaly.AnomalyDetector(self.series)
        detector.detrend(how='rolling')
        detector.fit()
        detector.plot()

