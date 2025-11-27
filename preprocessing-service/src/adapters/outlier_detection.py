"""
Adapters for outlier detection in time series.
"""
import pandas as pd
import numpy as np
from typing import List
from src.domain.ports import IOutlierDetector
from src.domain.models import TimeSeriesData, OutlierMethod


class StatisticalOutlierDetector(IOutlierDetector):
    """
    Outlier detector using statistical methods (Z-score, IQR).
    """
    
    def detect_and_remove(
        self, 
        data: TimeSeriesData, 
        method: OutlierMethod, 
        threshold: float
    ) -> TimeSeriesData:
        """Detect and remove outliers"""
        df = data.to_dataframe()
        
        if method == OutlierMethod.ZSCORE:
            z_scores = np.abs(
                (df['value'] - df['value'].mean()) / df['value'].std()
            )
            df = df[z_scores < threshold]
        
        elif method == OutlierMethod.IQR:
            Q1 = df['value'].quantile(0.25)
            Q3 = df['value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[
                (df['value'] >= lower_bound) & 
                (df['value'] <= upper_bound)
            ]
        
        elif method == OutlierMethod.ISOLATION_FOREST:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            values = df['value'].values.reshape(-1, 1)
            
            # Train isolation forest
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            predictions = iso_forest.fit_predict(values)
            
            # Keep only inliers (prediction == 1)
            df = df[predictions == 1]
        
        return TimeSeriesData.from_dataframe(df, data.metadata)
    
    def detect_only(
        self, 
        data: TimeSeriesData, 
        method: OutlierMethod, 
        threshold: float
    ) -> List[int]:
        """Detect outlier indices without removing"""
        df = data.to_dataframe()
        outlier_mask = np.zeros(len(df), dtype=bool)
        
        if method == OutlierMethod.ZSCORE:
            z_scores = np.abs(
                (df['value'] - df['value'].mean()) / df['value'].std()
            )
            outlier_mask = z_scores >= threshold
        
        elif method == OutlierMethod.IQR:
            Q1 = df['value'].quantile(0.25)
            Q3 = df['value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (df['value'] < lower_bound) | (df['value'] > upper_bound)
        
        elif method == OutlierMethod.ISOLATION_FOREST:
            from sklearn.ensemble import IsolationForest
            
            values = df['value'].values.reshape(-1, 1)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(values)
            outlier_mask = predictions == -1
        
        return np.where(outlier_mask)[0].tolist()