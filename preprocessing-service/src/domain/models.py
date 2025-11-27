"""
Domain models for the preprocessing service.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any
import pandas as pd


class InterpolationMethod(Enum):
    """Methods for handling missing values"""
    LINEAR = "linear"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    SPLINE = "spline"
    POLYNOMIAL = "polynomial"


class OutlierMethod(Enum):
    """Methods for detecting outliers"""
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"


class AggregationMethod(Enum):
    """Methods for aggregating resampled data"""
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


@dataclass
class TimeSeriesData:
    """
    Core domain model representing time series data.
    """
    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for processing"""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'value': self.values
        })
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, metadata: Dict[str, Any] = None):
        """Create from pandas DataFrame"""
        return cls(
            timestamps=df['timestamp'].tolist(),
            values=df['value'].tolist(),
            metadata=metadata or {}
        )
    
    def __len__(self):
        return len(self.values)


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing operations.
    Encapsulates all parameters needed for the preprocessing pipeline.
    """
    interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR
    outlier_method: OutlierMethod = OutlierMethod.ZSCORE
    outlier_threshold: float = 3.0
    resample_frequency: str = None
    aggregation_method: AggregationMethod = AggregationMethod.MEAN
    lag_features: List[int] = None
    rolling_window_sizes: List[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.outlier_threshold <= 0:
            raise ValueError("Outlier threshold must be positive")
        
        if self.lag_features and any(lag < 1 for lag in self.lag_features):
            raise ValueError("Lag values must be positive integers")
        
        if self.rolling_window_sizes and any(w < 2 for w in self.rolling_window_sizes):
            raise ValueError("Rolling window sizes must be at least 2")