"""
FastAPI application - REST API entry point.
"""
import sys
import os

# 1. Get the directory of this file (src/api/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up two levels to get the project root (preprocessing-service/)
project_root = os.path.dirname(os.path.dirname(current_dir))

# 3. Add the root to the system path
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

app = FastAPI(
    title="Data Preprocessing Service",
    description="Hexagonal architecture implementation for time series preprocessing",
    version="1.0.0"
)

# CORS middleware
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
    return {
        "service": "Data Preprocessing Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_series(
    request: PreprocessRequest,
    service: PreprocessingService = Depends(get_service)
):
    """
    Preprocess time series data.
    
    Applies missing value handling, outlier detection, and resampling.
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features", response_model=FeatureResponse)
async def create_features(
    request: FeatureRequest,
    service: PreprocessingService = Depends(get_service)
):
    """
    Create engineered features from time series.
    
    Generates lag features, rolling statistics, and time-based features.
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validate/{series_id}", response_model=ValidationResponse)
async def validate_series(
    series_id: str,
    service: PreprocessingService = Depends(get_service)
):
    """
    Validate time series data quality.
    
    Returns statistics about missing values, outliers, and data distribution.
    """
    try:
        validation = service.validate_data(series_id)
        return ValidationResponse(**validation)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)