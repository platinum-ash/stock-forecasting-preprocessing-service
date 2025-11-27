"""
Repository adapters for data persistence.
These implement the ITimeSeriesRepository port.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from src.domain.ports import ITimeSeriesRepository
from src.domain.models import TimeSeriesData


class TimescaleDBRepository(ITimeSeriesRepository):
    """
    TimescaleDB adapter for time series storage.
    In production, this would use SQLAlchemy or psycopg2.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # In production: 
        # from sqlalchemy import create_engine
        # self.engine = create_engine(connection_string)
        self._mock_data = {}  # For demo purposes
    
    def get_raw_data(self, series_id: str) -> TimeSeriesData:
        """Retrieve raw time series from database"""
        # Production SQL query:
        # query = """
        #     SELECT timestamp, value, metadata 
        #     FROM time_series_raw 
        #     WHERE series_id = %s 
        #     ORDER BY timestamp
        # """
        # df = pd.read_sql(query, self.engine, params=[series_id])
        
        # Mock implementation for demo
        if series_id in self._mock_data:
            return self._mock_data[series_id]
        
        # Generate mock data
        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
        values = np.random.randn(1000).cumsum() + 100
        
        # Add some missing values
        mask = np.random.random(1000) > 0.95
        values[mask] = np.nan
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values
        })
        
        data = TimeSeriesData.from_dataframe(df, {'series_id': series_id})
        self._mock_data[series_id] = data
        return data
    
    def save_preprocessed_data(self, series_id: str, data: TimeSeriesData) -> bool:
        """Save preprocessed data to database"""
        try:
            # Production implementation:
            # df = data.to_dataframe()
            # df['series_id'] = series_id
            # df.to_sql('time_series_preprocessed', self.engine, 
            #           if_exists='append', index=False)
            
            # Mock implementation
            self._mock_data[f"{series_id}_preprocessed"] = data
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def get_preprocessed_data(self, series_id: str) -> TimeSeriesData:
        """Retrieve preprocessed time series"""
        key = f"{series_id}_preprocessed"
        if key not in self._mock_data:
            raise ValueError(f"No preprocessed data found for {series_id}")
        return self._mock_data[key]


class InMemoryRepository(ITimeSeriesRepository):
    """
    Simple in-memory repository for testing.
    """
    
    def __init__(self):
        self._raw_data = {}
        self._preprocessed_data = {}
    
    def get_raw_data(self, series_id: str) -> TimeSeriesData:
        if series_id not in self._raw_data:
            raise ValueError(f"Series {series_id} not found")
        return self._raw_data[series_id]
    
    def save_preprocessed_data(self, series_id: str, data: TimeSeriesData) -> bool:
        self._preprocessed_data[series_id] = data
        return True
    
    def get_preprocessed_data(self, series_id: str) -> TimeSeriesData:
        if series_id not in self._preprocessed_data:
            raise ValueError(f"No preprocessed data for {series_id}")
        return self._preprocessed_data[series_id]
    
    def add_raw_data(self, series_id: str, data: TimeSeriesData):
        """Helper method for testing"""
        self._raw_data[series_id] = data