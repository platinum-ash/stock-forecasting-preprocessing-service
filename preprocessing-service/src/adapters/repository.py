"""
Repository adapters for data persistence.
These implement the ITimeSeriesRepository port.
"""

import pandas as pd
import json
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from typing import Optional
from src.domain.ports import ITimeSeriesRepository, ILogger
from src.domain.models import TimeSeriesData


class TimescaleDBRepository(ITimeSeriesRepository):
    """
    TimescaleDB adapter for time series storage using SQLAlchemy.
    Supports OHLCV (Open, High, Low, Close, Volume) data format.
    Uses separate databases for raw (ingestion) and preprocessed data.
    """

    def __init__(
        self, 
        ingestion_connection_string: str,
        preprocessing_connection_string: str,
        logger: ILogger
    ):
        """
        Args:
            ingestion_connection_string: Connection to ingestion database (read raw data)
            preprocessing_connection_string: Connection to preprocessing database (write preprocessed data)
            logger: Logger instance
        
        Example connection strings:
        postgresql+psycopg2://user:password@ingestion-timescaledb:5432/timeseries
        postgresql+psycopg2://user:password@preprocessing-timescaledb:5432/preprocessing
        """
        self.ingestion_engine = create_engine(ingestion_connection_string, pool_pre_ping=True)
        self.preprocessing_engine = create_engine(preprocessing_connection_string, pool_pre_ping=True)
        self.logger = logger

        # Ensure preprocessed schema exists (we don't modify ingestion DB)
        self._initialize_preprocessing_schema()

    def _initialize_preprocessing_schema(self):
        """Create preprocessed table + hypertable if they do not exist."""
        with self.preprocessing_engine.begin() as conn:
            # Enable TimescaleDB extension
            conn.execute(text("""
            CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
            """))

            # Create preprocessed data table
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS time_series_preprocessed (
                series_id TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                features JSONB DEFAULT '{}'::jsonb,
                PRIMARY KEY(series_id, timestamp)
            );
            """))

            # Create hypertable
            conn.execute(text("""
            SELECT create_hypertable('time_series_preprocessed', 'timestamp', if_not_exists => TRUE);
            """))

    # -------------------------------
    # READ RAW DATA (from ingestion DB)
    # -------------------------------
    def get_raw_data(self, series_id: str) -> TimeSeriesData:
        """
        Retrieve raw OHLCV data from ingestion TimescaleDB.
        """
        query = text("""
            SELECT timestamp, open, high, low, close, volume, features
            FROM time_series_raw
            WHERE series_id = :sid
            ORDER BY timestamp
        """)

        with self.ingestion_engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sid": series_id})

        if df.empty:
            self.logger.info(f"Query returned {len(df)} rows for series_id: '{series_id}'")
            # Check what series_ids actually exist
            with self.ingestion_engine.connect() as conn:
                existing = pd.read_sql(
                    text("SELECT DISTINCT series_id FROM time_series_raw LIMIT 10"), 
                    conn
                )
                self.logger.info(f"No data found. Available series_ids: {existing['series_id'].tolist()}")
            raise ValueError(f"No raw data found for {series_id}")

        # Parse JSONB features if they're strings
        if 'features' in df.columns:
            df['features'] = df['features'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x if x else {})
            )

        return TimeSeriesData.from_dataframe(
            df,
            metadata={"series_id": series_id}
        )

    # -------------------------------
    # SAVE RAW DATA (not used in preprocessing service)
    # -------------------------------
    def save_raw_data(self, series_id: str, data: TimeSeriesData) -> bool:
        """
        This method is not used in preprocessing service.
        Raw data is saved by the ingestion service.
        """
        self.logger.warning("save_raw_data called on preprocessing service - this is handled by ingestion service")
        return False

    # -------------------------------
    # SAVE PREPROCESSED DATA (to preprocessing DB)
    # -------------------------------
    def save_preprocessed_data(self, series_id: str, data: TimeSeriesData) -> bool:
        """
        Save preprocessed OHLCV data to preprocessing TimescaleDB.
        Features are stored as JSONB - one JSON object per timestamp.
        """
        try:
            df = data.to_dataframe()

            # Debug logging
            self.logger.info(f"Saving preprocessed data - DataFrame columns: {df.columns.tolist()}")
            if 'features' in df.columns and len(df) > 0:
                self.logger.info(f"First features value type: {type(df['features'].iloc[0])}")

            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Ensure features column exists
            if 'features' not in df.columns:
                df['features'] = [{}] * len(df)

            insert_stmt = text("""
            INSERT INTO time_series_preprocessed 
                (series_id, timestamp, open, high, low, close, volume, features)
            VALUES 
                (:series_id, :timestamp, :open, :high, :low, :close, :volume, :features)
            ON CONFLICT (series_id, timestamp) DO UPDATE
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                features = EXCLUDED.features;
            """)

            with self.preprocessing_engine.begin() as conn:
                rows = [
                    {
                        "series_id": series_id,
                        "timestamp": row["timestamp"],
                        "open": float(row["open"]) if pd.notna(row["open"]) else None,
                        "high": float(row["high"]) if pd.notna(row["high"]) else None,
                        "low": float(row["low"]) if pd.notna(row["low"]) else None,
                        "close": float(row["close"]) if pd.notna(row["close"]) else None,
                        "volume": float(row["volume"]) if pd.notna(row["volume"]) else None,
                        "features": json.dumps(row["features"]) if isinstance(row["features"], dict) else json.dumps({})
                    }
                    for _, row in df.iterrows()
                ]
                
                if rows:
                    self.logger.info(f"Inserting {len(rows)} rows for series {series_id}")
                
                conn.execute(insert_stmt, rows)

            return True

        except SQLAlchemyError as e:
            self.logger.error(f"Error saving preprocessed data: {e}")
            import traceback
            traceback.print_exc()
            return False

    # -------------------------------
    # READ PREPROCESSED DATA (from preprocessing DB)
    # -------------------------------
    def get_preprocessed_data(self, series_id: str) -> TimeSeriesData:
        """
        Retrieve preprocessed OHLCV data with features from preprocessing TimescaleDB.
        Features are parsed from JSONB back into Python dicts.
        """
        query = text("""
            SELECT timestamp, open, high, low, close, volume, features
            FROM time_series_preprocessed
            WHERE series_id = :sid
            ORDER BY timestamp
        """)

        with self.preprocessing_engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sid": series_id})

        if df.empty:
            raise ValueError(f"No preprocessed data found for {series_id}")
        
        # Parse JSONB features back to Python dicts (if they're strings)
        if 'features' in df.columns:
            df['features'] = df['features'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x if x else {})
            )

        return TimeSeriesData.from_dataframe(df, {"series_id": series_id})
    
    # -------------------------------
    # QUERY FEATURES
    # -------------------------------
    def get_feature_names(self, series_id: str, table: str = 'preprocessed') -> list:
        """
        Get all unique feature names for a series.
        
        Args:
            series_id: Time series identifier
            table: Either 'raw' or 'preprocessed'
        """
        engine = self.ingestion_engine if table == 'raw' else self.preprocessing_engine
        table_name = f"time_series_{table}"
        
        query = text(f"""
            SELECT DISTINCT jsonb_object_keys(features) as feature_name
            FROM {table_name}
            WHERE series_id = :sid
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"sid": series_id})
            return [row[0] for row in result]
    
    def get_data_with_specific_features(
        self, 
        series_id: str, 
        feature_names: list,
        table: str = 'preprocessed'
    ) -> pd.DataFrame:
        """
        Get OHLCV data with only specific features extracted.
        
        Args:
            series_id: Time series identifier
            feature_names: List of feature names to extract from JSONB
            table: Either 'raw' or 'preprocessed'
            
        Returns:
            DataFrame with timestamp, OHLCV columns, and selected features as columns
        """
        engine = self.ingestion_engine if table == 'raw' else self.preprocessing_engine
        table_name = f"time_series_{table}"
        
        # Build feature selection for each requested feature
        feature_selects = [
            f"features->'{name}' as {name}" 
            for name in feature_names
        ]
        feature_sql = ", ".join(feature_selects) if feature_selects else ""
        
        # Build column list
        base_cols = "timestamp, open, high, low, close, volume"
        select_cols = f"{base_cols}, {feature_sql}" if feature_sql else base_cols
        
        query = text(f"""
            SELECT {select_cols}
            FROM {table_name}
            WHERE series_id = :sid
            ORDER BY timestamp
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sid": series_id})
        
        return df
    
    # -------------------------------
    # UTILITY METHODS
    # -------------------------------
    def get_date_range(self, series_id: str, table: str = 'raw') -> tuple:
        """
        Get the earliest and latest timestamps for a series.
        
        Args:
            series_id: Time series identifier
            table: Either 'raw' or 'preprocessed'
            
        Returns:
            Tuple of (earliest_timestamp, latest_timestamp)
        """
        engine = self.ingestion_engine if table == 'raw' else self.preprocessing_engine
        table_name = f"time_series_{table}"
        
        query = text(f"""
            SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest
            FROM {table_name}
            WHERE series_id = :sid
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"sid": series_id}).fetchone()
            return (result[0], result[1]) if result else (None, None)
    
    def get_series_count(self, series_id: str, table: str = 'raw') -> int:
        """
        Get the number of records for a series.
        
        Args:
            series_id: Time series identifier
            table: Either 'raw' or 'preprocessed'
            
        Returns:
            Number of records
        """
        engine = self.ingestion_engine if table == 'raw' else self.preprocessing_engine
        table_name = f"time_series_{table}"
        
        query = text(f"""
            SELECT COUNT(*) as count
            FROM {table_name}
            WHERE series_id = :sid
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"sid": series_id}).fetchone()
            return result[0] if result else 0
