# Data Preprocessing Service

This service is part of a larger distributed system for collecting, analyzing, and forecasting time series data. Its primary responsibility is **preprocessing time series datasets** before they are used by downstream services like forecasting, anomaly detection, or analysis.

## Responsibilities

The preprocessing service handles:

- Outlier detection and removal
- Resampling and aggregation of time series
- Feature engineering (lag features, rolling statistics)

It serves as a key step to ensure high-quality, clean, and well-structured data for the rest of the system.

## Architecture

This service follows **hexagonal architecture**:

- **Domain Layer**: Pure business logic for preprocessing operations
- **Ports**: Interfaces defining contracts for adapters
- **Adapters**: Technology-specific implementations (e.g., Pandas, database connectors)
- **API Layer**: FastAPI REST endpoints to receive datasets and provide processed results


## Project Structure
```
preprocessing-service/
├── src/
│   ├── domain/          # Core business logic
│   ├── adapters/        # Infrastructure implementations
│   └── api/             # REST API
├── tests/               # Unit tests
├── Dockerfile 
└── requirements.txt

