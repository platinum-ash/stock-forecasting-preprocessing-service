"""
Core business logic for preprocessing time series data.
"""
from typing import Optional
import pandas as pd
from .models import TimeSeriesData, PreprocessingConfig
from .ports import (
    ITimeSeriesRepository,
    IMissingValueHandler,
    IOutlierDetector,
    IFeatureEngineer,
    IResampler,
    ILogger
)


class PreprocessingService:
    """
    Core preprocessing service implementing the business logic.
    Uses dependency injection - all dependencies are ports (interfaces).
    """
    
    def __init__(
        self,
        repository: ITimeSeriesRepository,
        missing_handler: IMissingValueHandler,
        outlier_detector: IOutlierDetector,
        feature_engineer: IFeatureEngineer,
        resampler: IResampler,
        logger: ILogger
    ):
        self.repository = repository
        self.missing_handler = missing_handler
        self.outlier_detector = outlier_detector
        self.feature_engineer = feature_engineer
        self.resampler = resampler
        self.logger = logger
    
    def preprocess(
        self, 
        series_id: str, 
        config: PreprocessingConfig
    ) -> TimeSeriesData:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            series_id: Unique identifier for the time series
            config: Preprocessing configuration
            
        Returns:
            Preprocessed time series data
        """
        try:
            self.logger.info(f"Starting preprocessing for series: {series_id}")
            
            # Step 1: Retrieve raw data
            data = self.repository.get_raw_data(series_id)
            original_count = len(data)
            self.logger.info(f"Retrieved {original_count} data points")
            
            # Step 2: Handle missing values
            data = self.missing_handler.handle_missing(
                data, 
                config.interpolation_method
            )
            self.logger.info(
                f"Missing values handled using {config.interpolation_method.value}"
            )
            
            # Step 3: Detect and remove outliers
            data = self.outlier_detector.detect_and_remove(
                data, 
                config.outlier_method, 
                config.outlier_threshold
            )
            removed_count = original_count - len(data)
            self.logger.info(
                f"Outliers processed: {removed_count} points removed "
                f"using {config.outlier_method.value}"
            )
            
            # Step 4: Resample if frequency specified
            if config.resample_frequency:
                data = self.resampler.resample(
                    data, 
                    config.resample_frequency, 
                    config.aggregation_method
                )
                self.logger.info(
                    f"Resampled to {config.resample_frequency} "
                    f"using {config.aggregation_method.value}"
                )
            
            # Step 5: Save preprocessed data
            success = self.repository.save_preprocessed_data(series_id, data)
            if success:
                self.logger.info("Preprocessing completed successfully")
            else:
                self.logger.warning("Failed to save preprocessed data")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed for {series_id}", e)
            raise
    
    def create_features(
        self, 
        series_id: str, 
        config: PreprocessingConfig
    ) -> pd.DataFrame:
        """
        Create engineered features from time series data.
        
        Args:
            series_id: Unique identifier for the time series
            config: Configuration with feature specifications
            
        Returns:
            DataFrame with original data and engineered features
        """
        try:
            self.logger.info(f"Creating features for series: {series_id}")
            
            # Get preprocessed data if available, otherwise raw data
            try:
                data = self.repository.get_preprocessed_data(series_id)
            except:
                data = self.repository.get_raw_data(series_id)
            
            df = data.to_dataframe()
            
            # Add lag features
            if config.lag_features:
                lag_df = self.feature_engineer.create_lag_features(
                    data, 
                    config.lag_features
                )
                df = pd.concat([df, lag_df], axis=1)
                self.logger.info(f"Created {len(config.lag_features)} lag features")
            
            # Add rolling features
            if config.rolling_window_sizes:
                rolling_df = self.feature_engineer.create_rolling_features(
                    data, 
                    config.rolling_window_sizes
                )
                df = pd.concat([df, rolling_df], axis=1)
                self.logger.info(
                    f"Created rolling features for {len(config.rolling_window_sizes)} windows"
                )
            
            # Add time-based features
            time_df = self.feature_engineer.create_time_features(data)
            df = pd.concat([df, time_df], axis=1)
            self.logger.info("Created time-based features")
            
            self.logger.info(f"Feature creation completed: {len(df.columns)} total features")
            return df
            
        except Exception as e:
            self.logger.error(f"Feature creation failed for {series_id}", e)
            raise
    
    def validate_data(self, series_id: str) -> dict:
        """
        Validate time series data quality.
        
        Returns:
            Dictionary with validation metrics
        """
        try:
            data = self.repository.get_raw_data(series_id)
            df = data.to_dataframe()
            
            validation = {
                'total_points': len(df),
                'missing_values': int(df['value'].isna().sum()),
                'missing_percentage': float((df['value'].isna().sum() / len(df)) * 100),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'value_stats': {
                    'mean': float(df['value'].mean()),
                    'std': float(df['value'].std()),
                    'min': float(df['value'].min()),
                    'max': float(df['value'].max())
                }
            }
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Validation failed for {series_id}", e)
            raise