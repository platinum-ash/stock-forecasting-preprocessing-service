"""
Dependency injection container for wiring adapters and services.
"""
import os
import logging
from typing import Optional

from src.domain.service import PreprocessingService
from src.domain.ports import IEventPublisher
from src.adapters.output.kafka import KafkaEventPublisher
from src.adapters.input.kafka import PreprocessingConsumer, IngestionEventHandler

logger = logging.getLogger(__name__)


class ApplicationContainer:
    """Container managing application dependencies"""
    
    def __init__(self):
        self._event_publisher: Optional[IEventPublisher] = None
        self._preprocessing_service: Optional[PreprocessingService] = None
        self._kafka_consumer: Optional[PreprocessingConsumer] = None
        self._bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    
    def get_event_publisher(self) -> IEventPublisher:
        """Get or create Kafka event publisher"""
        if self._event_publisher is None:
            self._event_publisher = KafkaEventPublisher(self._bootstrap_servers)
            logger.info("Kafka event publisher initialized")
        return self._event_publisher
    
    def get_preprocessing_service(self) -> PreprocessingService:
        """Get or create preprocessing service"""
        if self._preprocessing_service is None:
            # Import here to avoid circular dependencies
            from src.api.dependencies import get_service
            self._preprocessing_service = get_service()
            logger.info("Preprocessing service initialized")
        return self._preprocessing_service
    
    def get_kafka_consumer(self) -> PreprocessingConsumer:
        """Get or create Kafka consumer with wired dependencies"""
        if self._kafka_consumer is None:
            service = self.get_preprocessing_service()
            publisher = self.get_event_publisher()
            
            # Wire dependencies through the handler
            event_handler = IngestionEventHandler(service, publisher)
            self._kafka_consumer = PreprocessingConsumer(
                self._bootstrap_servers,
                event_handler
            )
            logger.info("Kafka consumer initialized")
        return self._kafka_consumer
    
    async def shutdown(self):
        """Clean up resources"""
        if self._kafka_consumer:
            await self._kafka_consumer.stop()
        if self._event_publisher:
            await self._event_publisher.close()
        logger.info("Application container shut down")


# Singleton instance
_container: Optional[ApplicationContainer] = None


def get_container() -> ApplicationContainer:
    """Get the application container singleton"""
    global _container
    if _container is None:
        _container = ApplicationContainer()
    return _container
