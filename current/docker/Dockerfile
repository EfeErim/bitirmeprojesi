# AADS-ULoRA v5.5.3-performance Production Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_optimized.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_optimized.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs adapters router config

# Set environment variables
ENV PYTHONPATH=/app/src
ENV APP_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]