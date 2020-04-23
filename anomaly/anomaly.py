import warnings
warnings.filterwarnings('ignore')
# mpl kicks up some warnings on MacOS
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seasonal


class AnomalyDetector:
    
    def __init__(self, series):
        """
        accepts a pandas series with DateTimeIndex
        of daily data.

        """
        self.series = series
        self.seasons = None
        self.trend = None
        self.adjusted = None
        self.residual = None
        self.anomaly = None
        self.indices = None
        self.trend_linear = None
        self.trend_rolling = None
        self.detrended = None

    
    def detrend(self, how='rolling', window=10):
        """
        Initially detrend the time-series removing either a linear
        regression line or a rolling median.

        self.detrended: the detrended data. 
        
        """
        if how == 'linear':
            lin = sp.stats.linregress(y=self.series, x=range(self.series.size))
            trend_linear = [(lin.slope * x) + lin.intercept for x in range(self.series.size)]
            self.trend_linear = pd.Series(trend_linear, index=self.series.index)
            self.detrended = self.series - self.trend_linear
        
        if how == 'rolling':
            roll = self.series.rolling(window=window)
            self.trend_rolling = roll.median().interpolate(method='time', limit_direction='both')
            self.detrended = self.series - self.trend_rolling                                                              
    
    def grubbs(self, array, func):
        """
        call the Grubb's outlier t-test on either minimum or maximum values
        recursively finds min or max outlier using a t-test, removes the outlier and iterates
        on the reduced data series until there are no more outliers left, keeps track of the
        removed indices.

        array:  data to find outliers in
        func:   either grubbs_min_index_g or grubbs_max_index_g

        """
        s = array[~np.isnan(array)].size
        index, g = func(array)
        significance = self.alpha / s
        self.t = sp.stats.t.isf(significance, s-2)
        g_test = ((s - 1) / np.sqrt(s)) * (np.sqrt(self.t**2 / (s - 2 + self.t**2)))
        if g > g_test:
            self.outliers.append(index)
            array[index] = np.nan
            self.grubbs(array, func)

    def grubbs_min_index_g(self, array):
        """
        function to perform z score on min outlier

        """
        index = np.nanargmin(array)
        g = abs(np.nanmin(array) - np.nanmean(array)) / np.nanstd(array)
        return index, g

    def grubbs_max_index_g(self, array):
        """
        function to perform z score on max outlier
         
        """
        index = np.nanargmax(array)
        g = abs(np.nanmax(array) - np.nanmean(array)) / np.nanstd(array)
        return index, g
            
    def fit(self, period=7, alpha=0.025):
        """
        first deseasonalize the data using holt-winters.

        fit using extreme z score with t value calculated from grubb's test.

        self.seasons:   the seasonal period as a short time-series
        self.trend:     the trend extracted by the holt-winters seasonal decomposition
        self.adjusted:  the time-series after seasonality extracted
        self.residual:  the time-series after trend and seasonality extracted
        self.outliers:  a time-series of Boolean values where anomalies are detected
        self.indices:   the indices of True anomaly values in self.outliers
        
        """
        self.alpha = alpha
        data = self.series if self.detrended is None else self.detrended

        if period is not None:
            trend_method = 'spline'
            self.seasons, trend = seasonal.fit_seasons(
                data, periodogram_thresh=0.5, trend=trend_method, period=period)
            self.trend = pd.Series(trend, index=self.series.index)
            if self.seasons is None:
                raise ValueError(
                    'period {} seasonality could not be extracted from the data'.format(period))
            self.adjusted = seasonal.adjust_seasons(
                data, seasons=self.seasons, trend=trend_method)
            self.residual = self.adjusted - self.trend

        self.residual = data if self.residual is None else self.residual
        self.outliers = []
        self.grubbs(self.residual.copy().astype(float).values, self.grubbs_min_index_g)
        self.grubbs(self.residual.copy().astype(float).values, self.grubbs_max_index_g)
        self.indices = np.array(self.outliers)
        self.anomaly = np.zeros(data.size, dtype=bool)
        self.anomaly[self.outliers] = True
        return self
        

    def plot(self):
        """
        helper function to plot the data
        
        """
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15, 10))
        xmin = self.series.index.min()
        xmax = self.series.index.max()
        
        ax0.plot(self.series.index, self.series, label='series')
        if self.detrended is not None:
            ax0.plot(self.series.index, self.detrended, label='initially detrended', color='green')
        if self.trend_linear is not None:
            ax0.plot(self.series.index, self.trend_linear, label='linear trend', color='red')
        if self.trend_rolling is not None:
            ax0.plot(self.series.index, self.trend_rolling, label='rolling trend', color='red')
        if self.adjusted is not None:
            ax0.plot(self.series.index, self.adjusted, label='seasonaly adjusted', color='orange')
        
        weekdays = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', None)
        colors = ('blue', 'green', 'orange', 'pink', 'purple', 'cyan', 'brown', 'violet')
        
        try:
            self.series.index.weekday
            for i, color in enumerate(colors):
                non_anomalous = self.series.where((~self.anomaly) & (self.series.index.weekday==i))
                ax1.scatter(self.series.index, non_anomalous, alpha=0.5, marker='.', color=color, label=weekdays[i])
        except AttributeError:
            non_anomalous = self.series.where(~self.anomaly)
            ax1.scatter(self.series.index, non_anomalous, alpha=0.5, marker='.', color='green', label='non anomalous')
            
        anomalous = self.series.where(self.anomaly)
        ax1.scatter(self.series.index, anomalous, alpha=0.8, marker='.', color='red', label='anomalies')

        ax2.plot(self.series.index, self.residual, color='blue', label='residual')

        ax0.legend(loc='upper left')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
        plt.show()


def detect_anomalies(series):
    """
    helper function to apply to series directly
    
    df = pd.DataFrame(...)
    
    df['anomaly'] = df['target'].apply(detect_anomaly)
    
    """
    detector = AnomalyDetector(series)
    detector.fit()
    return detector.anomaly

        

if __name__ == '__main__':
    import os
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(APP_ROOT, '..', 'tests', 'test.csv')
    series = pd.read_csv(csv_path, squeeze=True,
                         index_col=0, parse_dates=True)
    series[100] = 2
    series[150] = 170000
    detector = AnomalyDetector(series)
    detector.detrend(how='linear')
    detector.fit(period=7)
    detector.plot()
