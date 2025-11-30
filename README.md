# Data Preprocessing Service

Implementation for time series data preprocessing.

## Architecture

This service follows hexagonal architecture:

- **Domain Layer**: Pure business logic
- **Ports**: Interfaces defining contracts
- **Adapters**: Implementations using specific technologies
- **API Layer**: FastAPI REST endpoints

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