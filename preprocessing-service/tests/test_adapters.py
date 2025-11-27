"""
Tests for adapter implementations.
"""
import pytest
import pandas as pd
import numpy as np
from src.domain.models import TimeSeriesData, InterpolationMethod, OutlierMethod
from src.adapters.missing_values import PandasMissingValueHandler
from src.adapters.outlier_detection import StatisticalOutlierDetector


@pytest.fixture
def data_with_missing():
    """Create data with missing values"""
    dates = pd.date_range('2024-01-01', periods=50, freq='H')
    values = list(range(50))
    values[10:15] = [np.nan] * 5
    
    df = pd.DataFrame({'timestamp': dates, 'value': values})
    return TimeSeriesData.from_dataframe(df, {})


def test_linear_interpolation(data_with_missing):
    """Test linear interpolation of missing values"""
    handler = PandasMissingValueHandler()
    result = handler.handle_missing(data_with_missing, InterpolationMethod.LINEAR)
    
    df = result.to_dataframe()
    assert df['value'].isna().sum() == 0


def test_zscore_outlier_detection():
    """Test Z-score outlier detection"""
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    values = np.random.randn(100)
    values[50] = 10  # Add outlier
    
    df = pd.DataFrame({'timestamp': dates, 'value': values})
    data = TimeSeriesData.from_dataframe(df, {})
    
    detector = StatisticalOutlierDetector()
    result = detector.detect_and_remove(data, OutlierMethod.ZSCORE, 3.0)
    
    assert len(result) < len(data)