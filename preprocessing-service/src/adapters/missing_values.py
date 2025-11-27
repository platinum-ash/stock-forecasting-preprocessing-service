"""
Adapters for handling missing values in time series.
"""
import pandas as pd
import numpy as np
from src.domain.ports import IMissingValueHandler
from src.domain.models import TimeSeriesData, InterpolationMethod


class PandasMissingValueHandler(IMissingValueHandler):
    """
    Missing value handler using pandas interpolation methods.
    """
    
    def handle_missing(
        self, 
        data: TimeSeriesData, 
        method: InterpolationMethod
    ) -> TimeSeriesData:
        """Fill missing values using specified interpolation method"""
        df = data.to_dataframe()
        df.set_index('timestamp', inplace=True)
        
        if method == InterpolationMethod.LINEAR:
            df['value'] = df['value'].interpolate(method='linear')
        
        elif method == InterpolationMethod.FORWARD_FILL:
            df['value'] = df['value'].fillna(method='ffill')
        
        elif method == InterpolationMethod.BACKWARD_FILL:
            df['value'] = df['value'].fillna(method='bfill')
        
        elif method == InterpolationMethod.SPLINE:
            # Spline interpolation requires at least 4 points
            if len(df.dropna()) >= 4:
                df['value'] = df['value'].interpolate(method='spline', order=3)
            else:
                # Fall back to linear if not enough points
                df['value'] = df['value'].interpolate(method='linear')
        
        elif method == InterpolationMethod.POLYNOMIAL:
            if len(df.dropna()) >= 3:
                df['value'] = df['value'].interpolate(method='polynomial', order=2)
            else:
                df['value'] = df['value'].interpolate(method='linear')
        
        # Fill any remaining NaN at edges
        df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')
        
        df.reset_index(inplace=True)
        return TimeSeriesData.from_dataframe(df, data.metadata)