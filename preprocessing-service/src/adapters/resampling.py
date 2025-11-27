"""
Adapters for time series resampling.
"""
import pandas as pd
from src.domain.ports import IResampler
from src.domain.models import TimeSeriesData, AggregationMethod


class PandasResampler(IResampler):
    """
    Resampler using pandas resample functionality.
    """
    
    def resample(
        self, 
        data: TimeSeriesData, 
        frequency: str, 
        method: AggregationMethod
    ) -> TimeSeriesData:
        """Resample time series to different frequency"""
        df = data.to_dataframe()
        df.set_index('timestamp', inplace=True)
        
        agg_func = method.value
        df_resampled = df.resample(frequency).agg(agg_func)
        df_resampled.reset_index(inplace=True)
        
        # Remove any NaN values that might result from resampling
        df_resampled = df_resampled.dropna()
        
        return TimeSeriesData.from_dataframe(df_resampled, data.metadata)