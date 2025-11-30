"""
Ports (interfaces) defining the contracts between core business logic and adapters.
"""
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from .models import TimeSeriesData, InterpolationMethod, OutlierMethod, AggregationMethod


class ITimeSeriesRepository(ABC):
    """Port for time series data persistence"""
    
    @abstractmethod
    def get_raw_data(self, series_id: str) -> TimeSeriesData:
        """Retrieve raw time series data by ID"""
        pass
    
    @abstractmethod
    def save_preprocessed_data(self, series_id: str, data: TimeSeriesData) -> bool:
        """Save preprocessed time series data"""
        pass
    
    @abstractmethod
    def get_preprocessed_data(self, series_id: str) -> TimeSeriesData:
        """Retrieve preprocessed time series data by ID"""
        pass


class IMissingValueHandler(ABC):
    """Port for handling missing values in time series"""
    
    @abstractmethod
    def handle_missing(
        self, 
        data: TimeSeriesData, 
        method: InterpolationMethod
    ) -> TimeSeriesData:
        """Fill missing values using specified method"""
        pass


class IOutlierDetector(ABC):
    """Port for outlier detection and removal"""
    
    @abstractmethod
    def detect_and_remove(
        self, 
        data: TimeSeriesData, 
        method: OutlierMethod, 
        threshold: float
    ) -> TimeSeriesData:
        """Detect and remove outliers using specified method"""
        pass
    
    @abstractmethod
    def detect_only(
        self, 
        data: TimeSeriesData, 
        method: OutlierMethod, 
        threshold: float
    ) -> List[int]:
        """Detect outlier indices without removing them"""
        pass


class IFeatureEngineer(ABC):
    """Port for feature engineering operations"""
    
    @abstractmethod
    def create_lag_features(
        self, 
        data: TimeSeriesData, 
        lags: List[int]
    ) -> pd.DataFrame:
        """Create lagged features"""
        pass
    
    @abstractmethod
    def create_rolling_features(
        self, 
        data: TimeSeriesData, 
        windows: List[int]
    ) -> pd.DataFrame:
        """Create rolling window statistics features"""
        pass
    
    @abstractmethod
    def create_time_features(self, data: TimeSeriesData) -> pd.DataFrame:
        """Create time-based features (hour, day, month, etc.)"""
        pass


class IResampler(ABC):
    """Port for time series resampling"""
    
    @abstractmethod
    def resample(
        self, 
        data: TimeSeriesData, 
        frequency: str, 
        method: AggregationMethod
    ) -> TimeSeriesData:
        """Resample time series to different frequency"""
        pass


class ILogger(ABC):
    """Port for logging operations"""
    
    @abstractmethod
    def info(self, message: str):
        """Log informational message"""
        pass
    
    @abstractmethod
    def warning(self, message: str):
        """Log warning message"""
        pass
    
    @abstractmethod
    def error(self, message: str, exception: Exception = None):
        """Log error message with optional exception"""
        pass
    
    @abstractmethod
    def debug(self, message: str):
        """Log debug message"""
        pass