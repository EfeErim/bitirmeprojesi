# Contributing to AADS-ULoRA v5.5

Thank you for your interest in contributing to this project!

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure as needed
5. Install pre-commit hooks (if available)

## Project Structure

```
d:/bitirme projesi/
├── src/                    # Source code
│   ├── adapter/           # Crop-specific adapters
│   ├── dataset/           # Dataset handling
│   ├── debugging/         # Debugging utilities
│   ├── evaluation/        # Evaluation metrics
│   ├── ood/              # Out-of-distribution detection
│   ├── pipeline/         # Main pipeline orchestration
│   ├── router/           # Routing logic
│   ├── training/         # Training scripts
│   ├── utils/            # Utility functions
│   └── visualization/    # Visualization tools
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test fixtures
├── api/                   # FastAPI server
│   ├── endpoints/        # API endpoints
│   └── middleware/       # Middleware components
├── config/               # Configuration files
├── docs/                 # Documentation
│   ├── api/             # API documentation
│   ├── architecture/    # Architecture docs
│   ├── development/     # Development guides
│   ├── literature/      # Literature review
│   └── user_guide/      # User guides
├── scripts/              # Build and utility scripts
├── docker/               # Docker configuration
├── monitoring/           # Monitoring setup (Prometheus/Grafana)
├── mobile/               # Android mobile app
├── colab_notebooks/      # Google Colab notebooks
├── benchmarks/           # Performance benchmarks
└── logs/                # Application logs (created at runtime)
```

## Coding Standards

- Follow PEP 8 style guide for Python code
- Use type hints for function parameters and return values
- Write docstrings for all public functions, classes, and methods
- Keep functions small and focused on a single responsibility
- Write unit tests for new functionality

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/unit/
pytest tests/integration/
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commit messages
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request with a clear description of the changes

## Code Review

- All code changes require review
- Address review comments promptly
- Keep PRs focused on a single feature/fix

## Questions?

Feel free to open an issue for any questions or clarifications.