"""
Handles ingestion events and coordinates domain operations.
"""
import logging
from typing import Dict, Any
from src.domain.service import PreprocessingService
from src.domain.models import (
    PreprocessingConfig,
    InterpolationMethod,
    OutlierMethod,
    AggregationMethod
)
from src.domain.ports import IEventPublisher

logger = logging.getLogger(__name__)


class IngestionEventHandler:
    """Handles ingestion completion events and triggers preprocessing"""
    
    def __init__(
        self,
        preprocessing_service: PreprocessingService,
        event_publisher: IEventPublisher
    ):
        self.preprocessing_service = preprocessing_service
        self.event_publisher = event_publisher
    
    async def handle(self, event_data: Dict[str, Any]):
        """
        Process ingestion completion event.
        
        Args:
            event_data: Event payload containing series_id, job_id, and config
        """
        series_id = None
        job_id = None
        
        try:
            logger.info(f"Processing ingestion event: {event_data.get('series_id')}")
            
            series_id = event_data.get('series_id')
            job_id = event_data.get('job_id')
            config_data = event_data.get('preprocessing_config', {})
            
            if not series_id or not job_id:
                raise ValueError("Missing required fields: series_id or job_id")
            
            # Build preprocessing configuration
            config = self._build_config(config_data)
            
            # Execute preprocessing (domain logic)
            result = self.preprocessing_service.preprocess(series_id, config)
            
            # Create features
            features_df = self.preprocessing_service.create_features(series_id, config)
            
            # Publish success event
            await self.event_publisher.publish_preprocessing_completed(
                series_id=series_id,
                job_id=job_id,
                data_points=len(result.values),
                features_created=list(features_df.columns),
                metadata=result.metadata
            )
            
            logger.info(f"Successfully preprocessed series {series_id} for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing ingestion event: {e}", exc_info=True)
            
            # Publish failure event
            if series_id and job_id:
                await self.event_publisher.publish_processing_failed(
                    series_id=series_id,
                    job_id=job_id,
                    error=str(e),
                    stage='preprocessing'
                )
    
    def _build_config(self, config_data: Dict[str, Any]) -> PreprocessingConfig:
        """
        Build preprocessing config from event data with sensible defaults.
        
        Args:
            config_data: Configuration dictionary from event
            
        Returns:
            PreprocessingConfig object
        """
        return PreprocessingConfig(
            interpolation_method=InterpolationMethod(
                config_data.get('interpolation_method', 'linear')
            ),
            outlier_method=OutlierMethod(
                config_data.get('outlier_method', 'iqr')
            ),
            outlier_threshold=config_data.get('outlier_threshold', 3.0),
            resample_frequency=config_data.get('resample_frequency'),
            aggregation_method=AggregationMethod(
                config_data.get('aggregation_method', 'mean')
            ),
            lag_features=config_data.get('lag_features', [1, 7, 30]),
            rolling_window_sizes=config_data.get('rolling_window_sizes', [7, 30])
        )
