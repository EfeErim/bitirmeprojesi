# AADS-ULoRA v5.5 Deployment Guide

## Overview

This guide covers deployment options for the AADS-ULoRA v5.5 system, including local development, Docker, and production environments.

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for container deployment)
- Git
- (Optional) GPU with CUDA support for faster inference

## Local Development

### 1. Clone and Setup

```bash
git clone <repository-url>
cd bitirme-projesi
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` to configure:
- API host and port
- Model paths
- Logging settings
- Database configuration (if needed)

### 5. Run the API Server

```bash
python api/main.py
```

Or using Uvicorn directly:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Deployment

### 1. Build and Run with Docker Compose

```bash
docker-compose up --build
```

### 2. Environment Variables

Configure environment variables in `docker-compose.yml`:
```yaml
environment:
  - APP_ENV=production
  - API_HOST=0.0.0.0
  - API_PORT=8000
  - LOG_LEVEL=INFO
```

### 3. Production Configuration

For production, consider:
- Using a reverse proxy (Nginx)
- Enabling HTTPS
- Configuring proper logging
- Setting up monitoring

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Available Endpoints

- `GET /health` - Health check
- `POST /diagnose` - Disease diagnosis
- `GET /crops` - List supported crops
- `GET /metrics` - System metrics
- `GET /logs` - Recent logs (development only)

## Model Management

### Loading Models
Models should be placed in the configured model directory:
```
./models/
├── crop_adapters/
├── router_models/
└── base_models/
```

### Configuration

Model configuration is in `config/adapter_spec_v55.json`.

## Monitoring

### Prometheus Metrics
Metrics are available at `/metrics` endpoint.

### Grafana Dashboard
Setup Grafana with the provided provisioning configuration in `monitoring/grafana/`.

## Scaling

### Horizontal Scaling
For production deployments:
- Use multiple API workers
- Implement load balancing
- Consider using a process manager like PM2 or systemd

### Performance Optimization

- Enable GPU acceleration if available
- Configure appropriate cache sizes
- Optimize database queries (if using a database)
- Use connection pooling

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure models are in the correct directory
   - Check file permissions
   - Verify model compatibility

2. **Memory Issues**
   - Reduce batch sizes
   - Enable model quantization
   - Use smaller models for development

3. **API Connection Issues**
   - Check firewall settings
   - Verify port availability
   - Check CORS configuration

### Logs

Application logs are written to `logs/app.log` (configured in `.env`).

## Security Considerations

- Use HTTPS in production
- Implement proper authentication
- Validate all inputs
- Keep dependencies updated
- Regular security audits

## Backup and Recovery

- Regular database backups (if applicable)
- Model checkpointing
- Configuration backups
- Disaster recovery plan

## Support

For issues and questions:
1. Check the logs
2. Review the documentation
3. Search existing issues
4. Open a new issue with detailed information

## License

This project is licensed under the [insert license here] License.