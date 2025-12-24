"""
Input adapters for Kafka event consumption.
"""
from .consumer import PreprocessingConsumer
from .message_handler import IngestionEventHandler

__all__ = ['PreprocessingConsumer', 'IngestionEventHandler']
