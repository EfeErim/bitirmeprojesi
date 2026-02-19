# Deployment Guide

## Current Deployment Modes

### 1) Local API Runtime

```powershell
$env:APP_ENV="production"
python -m api.main
```

### 2) Container-Based Runtime

Use files under `docker/`:

- `docker/Dockerfile`
- `docker/docker-compose.yml`

## Pre-Deployment Checklist

1. Validate config files in `config/`.
2. Ensure API starts without exceptions.
3. Run API and integration tests relevant to the release.
4. Confirm required model/checkpoint artifacts are available.

## Health and Monitoring Endpoints

- `/health`
- `/v1/liveness`
- `/v1/readiness`
- `/v1/metrics`

## Rollback

See `docs/development/rollback-guide.md` for rollback procedure.
