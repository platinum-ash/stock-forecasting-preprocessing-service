"""
Tests for adapter implementations.
"""
"""
Test cases for FeatureEngineer adapter.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.adapters.feature_engineering import FeatureEngineer
from src.domain.models import TimeSeriesData


class TestFeatureEngineerInitialization:
    """Test initialization of FeatureEngineer."""
    
    def test_default_initialization(self):
        fe = FeatureEngineer()
        assert fe.price_column == 'close'
    
    @pytest.mark.parametrize("price_col", ['open', 'high', 'low', 'close'])
    def test_initialization_with_different_price_columns(self, price_col):
        fe = FeatureEngineer(price_column=price_col)
        assert fe.price_column == price_col


class TestCreateLagFeatures:
    """Test lag feature creation."""
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample OHLCV time series data."""
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(95, 115, 100),
            'low': np.random.uniform(85, 105, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        return TimeSeriesData.from_dataframe(df)
    
    def test_create_lag_features_single_lag(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='close')
        lag_features = fe.create_lag_features(sample_time_series_data, lags=[1])
        
        assert 'lag_1' in lag_features.columns
        assert len(lag_features) == len(sample_time_series_data.to_dataframe())
        assert pd.isna(lag_features.iloc[0]['lag_1'])
    
    def test_create_lag_features_multiple_lags(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='close')
        lags = [1, 7, 30]
        lag_features = fe.create_lag_features(sample_time_series_data, lags=lags)
        
        assert all(f'lag_{lag}' in lag_features.columns for lag in lags)
        assert len(lag_features.columns) == len(lags)
    
    def test_create_lag_features_with_different_price_column(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='open')
        lag_features = fe.create_lag_features(sample_time_series_data, lags=[1])
        
        df = sample_time_series_data.to_dataframe()
        assert lag_features.iloc[1]['lag_1'] == df.iloc[0]['open']
    
    def test_create_lag_features_invalid_price_column(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='invalid_column')
        
        with pytest.raises(ValueError, match="Price column 'invalid_column' not found"):
            fe.create_lag_features(sample_time_series_data, lags=[1])
    
    def test_create_lag_features_correct_shift_values(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='close')
        lag_features = fe.create_lag_features(sample_time_series_data, lags=[1, 7])
        
        df = sample_time_series_data.to_dataframe()
        assert lag_features.iloc[7]['lag_7'] == df.iloc[0]['close']
        assert lag_features.iloc[10]['lag_1'] == df.iloc[9]['close']


class TestCreateRollingFeatures:
    """Test rolling window feature creation."""
    
    @pytest.fixture
    def sample_time_series_data(self):
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.arange(100) + 100,
            'high': np.arange(100) + 105,
            'low': np.arange(100) + 95,
            'close': np.arange(100) + 100,
            'volume': np.arange(100) * 10
        })
        return TimeSeriesData.from_dataframe(df)
    
    def test_create_rolling_features_single_window(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='close')
        rolling_features = fe.create_rolling_features(sample_time_series_data, windows=[7])
        
        expected_columns = ['rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7']
        assert all(col in rolling_features.columns for col in expected_columns)
    
    def test_create_rolling_features_multiple_windows(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='close')
        windows = [7, 30]
        rolling_features = fe.create_rolling_features(sample_time_series_data, windows=windows)
        
        assert len(rolling_features.columns) == len(windows) * 4
    
    def test_create_rolling_features_mean_calculation(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='close')
        rolling_features = fe.create_rolling_features(sample_time_series_data, windows=[7])
        
        df = sample_time_series_data.to_dataframe()
        expected_mean = df['close'].iloc[0:7].mean()
        assert rolling_features.iloc[6]['rolling_mean_7'] == expected_mean
    
    def test_create_rolling_features_nan_handling(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='close')
        rolling_features = fe.create_rolling_features(sample_time_series_data, windows=[7])
        
        assert pd.isna(rolling_features.iloc[0]['rolling_mean_7'])
        assert not pd.isna(rolling_features.iloc[6]['rolling_mean_7'])
    
    def test_create_rolling_features_invalid_price_column(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='nonexistent')
        
        with pytest.raises(ValueError, match="Price column 'nonexistent' not found"):
            fe.create_rolling_features(sample_time_series_data, windows=[7])
    
    def test_create_rolling_features_std_min_max(self, sample_time_series_data):
        fe = FeatureEngineer(price_column='close')
        rolling_features = fe.create_rolling_features(sample_time_series_data, windows=[10])
        
        df = sample_time_series_data.to_dataframe()
        window_data = df['close'].iloc[0:10]
        
        assert rolling_features.iloc[9]['rolling_std_10'] == window_data.std()
        assert rolling_features.iloc[9]['rolling_min_10'] == window_data.min()
        assert rolling_features.iloc[9]['rolling_max_10'] == window_data.max()


class TestCreateTimeFeatures:
    """Test time-based feature creation."""
    
    @pytest.fixture
    def sample_time_series_data(self):
        dates = pd.date_range(start='2025-01-01 00:00:00', periods=200, freq='h')

        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(90, 110, 200),
            'high': np.random.uniform(95, 115, 200),
            'low': np.random.uniform(85, 105, 200),
            'close': np.random.uniform(90, 110, 200),
            'volume': np.random.uniform(1000, 5000, 200)
        })
        return TimeSeriesData.from_dataframe(df)
    
    def test_create_time_features_column_names(self, sample_time_series_data):
        fe = FeatureEngineer()
        time_features = fe.create_time_features(sample_time_series_data)
        
        expected_columns = [
            'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
            'is_weekend', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'
        ]
        assert all(col in time_features.columns for col in expected_columns)
    
    def test_create_time_features_hour_extraction(self, sample_time_series_data):
        fe = FeatureEngineer()
        time_features = fe.create_time_features(sample_time_series_data)
        
        assert time_features.iloc[0]['hour'] == 0
        assert time_features.iloc[5]['hour'] == 5
    
    def test_create_time_features_weekend_detection(self):
        dates = pd.date_range(start='2025-01-04', periods=7, freq='D')  # Saturday start
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100] * 7,
            'high': [105] * 7,
            'low': [95] * 7,
            'close': [100] * 7,
            'volume': [1000] * 7
        })
        ts_data = TimeSeriesData.from_dataframe(df)
        fe = FeatureEngineer()
        time_features = fe.create_time_features(ts_data)
        
        assert time_features.iloc[0]['is_weekend'] == 1  # Saturday
        assert time_features.iloc[1]['is_weekend'] == 1  # Sunday
        assert time_features.iloc[2]['is_weekend'] == 0  # Monday
    
    def test_create_time_features_cyclical_encoding(self, sample_time_series_data):
        fe = FeatureEngineer()
        time_features = fe.create_time_features(sample_time_series_data)
        
        assert -1 <= time_features['month_sin'].max() <= 1
        assert -1 <= time_features['month_cos'].max() <= 1
        assert -1 <= time_features['day_of_week_sin'].max() <= 1
        assert -1 <= time_features['day_of_week_cos'].max() <= 1
    
    def test_create_time_features_quarter_and_year(self):
        dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100] * 365,
            'high': [105] * 365,
            'low': [95] * 365,
            'close': [100] * 365,
            'volume': [1000] * 365
        })
        ts_data = TimeSeriesData.from_dataframe(df)
        fe = FeatureEngineer()
        time_features = fe.create_time_features(ts_data)
        
        assert time_features.iloc[0]['quarter'] == 1
        assert time_features.iloc[0]['year'] == 2025
        assert time_features.iloc[180]['quarter'] in [2, 3]


class TestCreateOHLCVFeatures:
    """Test OHLCV-specific technical feature creation."""
    
    @pytest.fixture
    def sample_time_series_data(self):
        dates = pd.date_range(start='2025-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100, 102, 101, 103, 99] * 10,
            'high': [105, 107, 106, 108, 104] * 10,
            'low': [95, 97, 96, 98, 94] * 10,
            'close': [102, 101, 103, 99, 100] * 10,
            'volume': [1000, 1500, 1200, 1800, 1100] * 10
        })
        return TimeSeriesData.from_dataframe(df)
    
    def test_create_ohlcv_features_column_names(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        expected_columns = [
            'price_range', 'price_range_pct', 'body', 'body_pct',
            'upper_wick', 'lower_wick', 'close_position', 'vwap',
            'typical_price', 'close_change', 'close_pct_change',
            'volume_change', 'volume_pct_change', 'volume_price_trend',
            'true_range', 'gap', 'gap_pct'
        ]
        assert all(col in ohlcv_features.columns for col in expected_columns)
    
    def test_create_ohlcv_features_price_range_calculation(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        df = sample_time_series_data.to_dataframe()
        expected_range = df.iloc[0]['high'] - df.iloc[0]['low']
        assert ohlcv_features.iloc[0]['price_range'] == expected_range
    
    def test_create_ohlcv_features_body_calculation(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        df = sample_time_series_data.to_dataframe()
        expected_body = df.iloc[0]['close'] - df.iloc[0]['open']
        assert ohlcv_features.iloc[0]['body'] == expected_body
    
    def test_create_ohlcv_features_close_position(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        assert all((ohlcv_features['close_position'] >= 0) & 
                   (ohlcv_features['close_position'] <= 1))
    
    def test_create_ohlcv_features_close_position_zero_range(self):
        dates = pd.date_range(start='2025-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100] * 5,
            'high': [100] * 5,  # Same as low - zero range
            'low': [100] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5
        })
        ts_data = TimeSeriesData.from_dataframe(df)
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(ts_data)
        
        assert ohlcv_features.iloc[0]['close_position'] == 0.5
    
    def test_create_ohlcv_features_typical_price(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        df = sample_time_series_data.to_dataframe()
        expected_typical = (df.iloc[0]['high'] + df.iloc[0]['low'] + df.iloc[0]['close']) / 3
        assert ohlcv_features.iloc[0]['typical_price'] == expected_typical
    
    def test_create_ohlcv_features_vwap_calculation(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        df = sample_time_series_data.to_dataframe()
        typical_price = (df.iloc[0]['high'] + df.iloc[0]['low'] + df.iloc[0]['close']) / 3
        expected_vwap = typical_price * df.iloc[0]['volume']
        assert ohlcv_features.iloc[0]['vwap'] == expected_vwap
    
    def test_create_ohlcv_features_price_change(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        assert pd.isna(ohlcv_features.iloc[0]['close_change'])
        assert pd.isna(ohlcv_features.iloc[0]['close_pct_change'])
        assert not pd.isna(ohlcv_features.iloc[1]['close_change'])
    
    def test_create_ohlcv_features_volume_change(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        df = sample_time_series_data.to_dataframe()
        expected_vol_change = df.iloc[1]['volume'] - df.iloc[0]['volume']
        assert ohlcv_features.iloc[1]['volume_change'] == expected_vol_change
    
    def test_create_ohlcv_features_true_range(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        df = sample_time_series_data.to_dataframe()
        simple_range = df.iloc[0]['high'] - df.iloc[0]['low']
        assert ohlcv_features.iloc[0]['true_range'] == simple_range
    
    def test_create_ohlcv_features_gap_detection(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        df = sample_time_series_data.to_dataframe()
        expected_gap = df.iloc[1]['open'] - df.iloc[0]['close']
        assert ohlcv_features.iloc[1]['gap'] == expected_gap
    
    def test_create_ohlcv_features_wick_calculation(self, sample_time_series_data):
        fe = FeatureEngineer()
        ohlcv_features = fe.create_ohlcv_features(sample_time_series_data)
        
        assert all(ohlcv_features['upper_wick'] >= 0)
        assert all(ohlcv_features['lower_wick'] >= 0)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ts_data = TimeSeriesData.from_dataframe(df)
        fe = FeatureEngineer()
        
        lag_features = fe.create_lag_features(ts_data, lags=[1])
        assert len(lag_features) == 0
    
    def test_single_row_dataframe(self):
        df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-01-01')],
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000]
        })
        ts_data = TimeSeriesData.from_dataframe(df)
        fe = FeatureEngineer()
        
        lag_features = fe.create_lag_features(ts_data, lags=[1])
        assert len(lag_features) == 1
        assert pd.isna(lag_features.iloc[0]['lag_1'])
    
    def test_nan_values_in_data(self):
        dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100, np.nan, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000] * 10
        })
        ts_data = TimeSeriesData.from_dataframe(df)
        fe = FeatureEngineer(price_column='close')
        
        rolling_features = fe.create_rolling_features(ts_data, windows=[3])
        assert not rolling_features.empty
    
    @pytest.mark.parametrize("lags", [[], [0], [-1], [1, 2, 3]])
    def test_various_lag_configurations(self, lags):
        dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': range(10),
            'high': range(10, 20),
            'low': range(0, 10),
            'close': range(5, 15),
            'volume': [1000] * 10
        })
        ts_data = TimeSeriesData.from_dataframe(df)
        fe = FeatureEngineer()
        
        lag_features = fe.create_lag_features(ts_data, lags=lags)
        assert len(lag_features.columns) == len(lags)


class TestIndexPreservation:
    """Test that indices are preserved correctly."""
    
    def test_lag_features_index_preservation(self):
        dates = pd.date_range(start='2025-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': range(50),
            'high': range(50, 100),
            'low': range(0, 50),
            'close': range(25, 75),
            'volume': [1000] * 50
        })
        ts_data = TimeSeriesData.from_dataframe(df)
        fe = FeatureEngineer()
        
        lag_features = fe.create_lag_features(ts_data, lags=[1, 7])
        original_df = ts_data.to_dataframe()
        
        pd.testing.assert_index_equal(lag_features.index, original_df.index)
    
    def test_rolling_features_index_preservation(self):
        dates = pd.date_range(start='2025-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': range(50),
            'high': range(50, 100),
            'low': range(0, 50),
            'close': range(25, 75),
            'volume': [1000] * 50
        })
        ts_data = TimeSeriesData.from_dataframe(df)
        fe = FeatureEngineer()
        
        rolling_features = fe.create_rolling_features(ts_data, windows=[7, 30])
        original_df = ts_data.to_dataframe()
        
        pd.testing.assert_index_equal(rolling_features.index, original_df.index)
