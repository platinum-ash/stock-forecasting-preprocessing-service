"""
FastAPI application - REST API entry point.
Pure adapter, no knowledge of other adapters.
"""
import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    PreprocessRequest,
    FeatureRequest,
    PreprocessResponse,
    FeatureResponse,
    ValidationResponse
)
from src.api.dependencies import get_service
from src.domain.service import PreprocessingService
from src.domain.models import (
    PreprocessingConfig,
    InterpolationMethod,
    OutlierMethod,
    AggregationMethod
)
from src.application.container import get_container

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting application...")
    container = get_container()
    
    try:
        # Start Kafka consumer in background
        consumer = container.get_kafka_consumer()
        asyncio.create_task(consumer.start())
        
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    try:
        await container.shutdown()
        logger.info("Application shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


app = FastAPI(
    title="Data Preprocessing Service",
    description="Hexagonal architecture implementation for time series preprocessing",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    container = get_container()
    consumer = container.get_kafka_consumer()
    
    return {
        "service": "Data Preprocessing Service",
        "status": "running",
        "version": "1.0.0",
        "kafka_consumer_running": consumer.is_running if consumer else False
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    container = get_container()
    consumer = container.get_kafka_consumer()
    
    return {
        "status": "healthy",
        "components": {
            "api": "running",
            "kafka_consumer": "running" if (consumer and consumer.is_running) else "stopped",
            "kafka_producer": "connected"
        }
    }


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_series(
    request: PreprocessRequest,
    service: PreprocessingService = Depends(get_service)
):
    """
    Preprocess time series data (synchronous REST endpoint).
    """
    try:
        config = PreprocessingConfig(
            interpolation_method=InterpolationMethod(request.interpolation_method),
            outlier_method=OutlierMethod(request.outlier_method),
            outlier_threshold=request.outlier_threshold,
            resample_frequency=request.resample_frequency,
            aggregation_method=AggregationMethod(request.aggregation_method)
        )
        
        result = service.preprocess(request.series_id, config)
        
        return PreprocessResponse(
            status="success",
            series_id=request.series_id,
            data_points=len(result.values),
            metadata=result.metadata
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features", response_model=FeatureResponse)
async def create_features(
    request: FeatureRequest,
    service: PreprocessingService = Depends(get_service)
):
    """
    Create engineered features from time series.
    """
    try:
        config = PreprocessingConfig(
            lag_features=request.lag_features,
            rolling_window_sizes=request.rolling_window_sizes
        )
        
        features_df = service.create_features(request.series_id, config)
        
        return FeatureResponse(
            status="success",
            series_id=request.series_id,
            features=list(features_df.columns),
            rows=len(features_df)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Feature creation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validate/{series_id}", response_model=ValidationResponse)
async def validate_series(
    series_id: str,
    service: PreprocessingService = Depends(get_service)
):
    """
    Validate time series data quality.
    """
    try:
        validation = service.validate_data(series_id)
        return ValidationResponse(**validation)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
