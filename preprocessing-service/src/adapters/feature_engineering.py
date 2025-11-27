"""
Adapters for feature engineering operations.
"""
import pandas as pd
import numpy as np
from typing import List
from src.domain.ports import IFeatureEngineer
from src.domain.models import TimeSeriesData


class PandasFeatureEngineer(IFeatureEngineer):
    """
    Feature engineer using pandas for creating lag and rolling features.
    """
    
    def create_lag_features(
        self, 
        data: TimeSeriesData, 
        lags: List[int]
    ) -> pd.DataFrame:
        """Create lagged features"""
        df = data.to_dataframe()
        lag_features = pd.DataFrame(index=df.index)
        
        for lag in lags:
            lag_features[f'lag_{lag}'] = df['value'].shift(lag)
        
        return lag_features
    
    def create_rolling_features(
        self, 
        data: TimeSeriesData, 
        windows: List[int]
    ) -> pd.DataFrame:
        """Create rolling window statistics features"""
        df = data.to_dataframe()
        rolling_features = pd.DataFrame(index=df.index)
        
        for window in windows:
            rolling = df['value'].rolling(window=window)
            rolling_features[f'rolling_mean_{window}'] = rolling.mean()
            rolling_features[f'rolling_std_{window}'] = rolling.std()
            rolling_features[f'rolling_min_{window}'] = rolling.min()
            rolling_features[f'rolling_max_{window}'] = rolling.max()
        
        return rolling_features
    
    def create_time_features(self, data: TimeSeriesData) -> pd.DataFrame:
        """Create time-based features (hour, day, month, etc.)"""
        df = data.to_dataframe()
        time_features = pd.DataFrame(index=df.index)
        
        timestamps = pd.to_datetime(df['timestamp'])
        
        time_features['hour'] = timestamps.dt.hour
        time_features['day_of_week'] = timestamps.dt.dayofweek
        time_features['day_of_month'] = timestamps.dt.day
        time_features['month'] = timestamps.dt.month
        time_features['quarter'] = timestamps.dt.quarter
        time_features['year'] = timestamps.dt.year
        time_features['is_weekend'] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
        
        return time_features