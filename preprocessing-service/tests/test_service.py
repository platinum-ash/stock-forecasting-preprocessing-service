"""
Unit tests for the preprocessing service.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.domain.service import PreprocessingService
from src.domain.models import (
    TimeSeriesData,
    PreprocessingConfig,
    InterpolationMethod,
    OutlierMethod,
    AggregationMethod
)
from src.adapters.repository import InMemoryRepository
from src.adapters.missing_values import PandasMissingValueHandler
from src.adapters.outlier_detection import StatisticalOutlierDetector
from src.adapters.feature_engineering import PandasFeatureEngineer
from src.adapters.resampling import PandasResampler
from src.adapters.logging import ConsoleLogger


@pytest.fixture
def sample_data():
    """Create sample time series data for testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    values = np.random.randn(100).cumsum() + 100
    
    # Add some missing values
    values[10:15] = np.nan
    
    # Add some outliers
    values[50] = values.mean() + 10 * values.std()
    
    df = pd.DataFrame({
        'timestamp': dates,
        'value': values
    })
    
    return TimeSeriesData.from_dataframe(df, {'test': True})


@pytest.fixture
def service():
    """Create preprocessing service with test dependencies"""
    repository = InMemoryRepository()
    missing_handler = PandasMissingValueHandler()
    outlier_detector = StatisticalOutlierDetector()
    feature_engineer = PandasFeatureEngineer()
    resampler = PandasResampler()
    logger = ConsoleLogger()
    
    return PreprocessingService(
        repository=repository,
        missing_handler=missing_handler,
        outlier_detector=outlier_detector,
        feature_engineer=feature_engineer,
        resampler=resampler,
        logger=logger
    )


def test_preprocess_pipeline(service, sample_data):
    """Test complete preprocessing pipeline"""
    # Add data to repository
    service.repository.add_raw_data('test_series', sample_data)
    
    config = PreprocessingConfig(
        interpolation_method=InterpolationMethod.LINEAR,
        outlier_method=OutlierMethod.ZSCORE,
        outlier_threshold=3.0
    )
    
    result = service.preprocess('test_series', config)
    
    assert len(result) > 0
    assert len(result) <= len(sample_data)


def test_create_features(service, sample_data):
    """Test feature engineering"""
    service.repository.add_raw_data('test_series', sample_data)
    
    config = PreprocessingConfig(
        lag_features=[1, 7],
        rolling_window_sizes=[7]
    )
    
    features_df = service.create_features('test_series', config)
    
    assert 'lag_1' in features_df.columns
    assert 'lag_7' in features_df.columns
    assert 'rolling_mean_7' in features_df.columns


def test_validate_data(service, sample_data):
    """Test data validation"""
    service.repository.add_raw_data('test_series', sample_data)
    
    validation = service.validate_data('test_series')
    
    assert 'total_points' in validation
    assert 'missing_values' in validation
    assert validation['total_points'] == len(sample_data)