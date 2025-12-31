"""
Kafka producer - Output adapter for publishing events.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from aiokafka import AIOKafkaProducer
from src.domain.ports import IEventPublisher

logger = logging.getLogger(__name__)


class KafkaEventPublisher(IEventPublisher):
    """Kafka implementation of event publisher port"""
    
    def __init__(
        self,
        bootstrap_servers: str,
        completed_topic: str = 'data.preprocessing.completed',
        failed_topic: str = 'data.processing.failed'
    ):
        self.bootstrap_servers = bootstrap_servers
        self.completed_topic = os.getenv("KAFKA_OUTPUT_TOPIC", completed_topic)
        self.failed_topic = os.getenv("KAFKA_ERR_TOPIC", failed_topic)
        self.producer: Optional[AIOKafkaProducer] = None
    
    async def _get_producer(self) -> AIOKafkaProducer:
        """Lazy initialization of producer"""
        if self.producer is None:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip',
                max_request_size=1048576  # 1MB
            )
            await self.producer.start()
            logger.info(f"Kafka producer started: {self.bootstrap_servers}")
        return self.producer
    
    async def publish_preprocessing_completed(
        self,
        series_id: str,
        job_id: str,
        data_points: int,
        features_created: List[str],
        metadata: Dict[str, Any]
    ):
        """Publish preprocessing completion event"""
        event = {
            'event_type': 'preprocessing.completed',
            'timestamp': datetime.utcnow().isoformat(),
            'series_id': series_id,
            'job_id': job_id,
            'data_points': data_points,
            'features_created': features_created,
            'metadata': metadata
        }
        
        try:
            producer = await self._get_producer()
            await producer.send(self.completed_topic, value=event)
            logger.info(
                f"Published preprocessing completed event - "
                f"Job: {job_id}, Series: {series_id}, Points: {data_points}"
            )
        except Exception as e:
            logger.error(f"Failed to publish completion event: {e}", exc_info=True)
            raise
    
    async def publish_processing_failed(
        self,
        series_id: str,
        job_id: str,
        error: str,
        stage: str
    ):
        """Publish processing failure event"""
        event = {
            'event_type': 'processing.failed',
            'timestamp': datetime.utcnow().isoformat(),
            'series_id': series_id,
            'job_id': job_id,
            'stage': stage,
            'error': error
        }
        
        try:
            producer = await self._get_producer()
            await producer.send(self.failed_topic, value=event)
            logger.error(
                f"Published processing failed event - "
                f"Job: {job_id}, Series: {series_id}, Stage: {stage}, Error: {error}"
            )
        except Exception as e:
            logger.error(f"Failed to publish failure event: {e}", exc_info=True)
    
    async def close(self):
        """Close producer connection"""
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer closed")
