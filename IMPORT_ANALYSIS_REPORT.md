# Import and Dependency Analysis Report

## Summary

- Total Python files analyzed: 46
- Circular dependency cycles: 0
- Total unused imports: 65
- Directories missing __init__.py: 16
- Directories with mixed import styles: 1

## 1. Missing __init__.py Files

The following directories contain Python files but lack `__init__.py`:

- `api\endpoints`
- `api\middleware`
- `src\adapter`
- `src\core`
- `src\dataset`
- `src\debugging`
- `src\evaluation`
- `src\middleware`
- `src\monitoring`
- `src\ood`
- `src\pipeline`
- `src\router`
- `src\security`
- `src\training`
- `src\utils`
- `src\visualization`

**Recommendation:** Add empty `__init__.py` files to these directories to ensure proper package structure.

## 2. Circular Dependencies

No circular dependencies detected.

## 3. Unused Imports

The following files have unused imports:

### api\endpoints\crops.py

| Line | Module | Name |
|------|--------|------|
| 2 | `pydantic` | `validator` |
| 3 | `typing` | `Optional` |

### api\endpoints\diagnose.py

| Line | Module | Name |
|------|--------|------|
| 1 | `fastapi` | `Depends` |
| 6 | `pathlib` | `Path` |
| 10 | `torch` | `torch` |

### api\endpoints\feedback.py

| Line | Module | Name |
|------|--------|------|
| 1 | `fastapi` | `HTTPException` |
| 4 | `sys` | `sys` |
| 5 | `os` | `os` |
| 6 | `uuid` | `uuid` |

### api\endpoints\monitoring.py

| Line | Module | Name |
|------|--------|------|
| 10 | `typing` | `Dict` |
| 10 | `typing` | `Any` |

### api\graceful_shutdown.py

| Line | Module | Name |
|------|--------|------|
| 6 | `sys` | `sys` |

### api\main.py

| Line | Module | Name |
|------|--------|------|
| 8 | `sys` | `sys` |
| 10 | `pathlib` | `Path` |
| 11 | `json` | `json` |
| 12 | `asyncio` | `asyncio` |
| 15 | `fastapi.responses` | `Response` |

### api\metrics.py

| Line | Module | Name |
|------|--------|------|
| 7 | `src.monitoring.metrics` | `metrics_collector` |

### api\middleware\audit.py

| Line | Module | Name |
|------|--------|------|
| 6 | `src.middleware.audit` | `AuditMiddleware` |
| 6 | `src.middleware.audit` | `AuditLogger` |

### api\middleware\auth.py

| Line | Module | Name |
|------|--------|------|
| 6 | `src.middleware.auth` | `APIKeyMiddleware` |

### api\middleware\caching.py

| Line | Module | Name |
|------|--------|------|
| 6 | `src.middleware.caching` | `CacheMiddleware` |
| 6 | `src.middleware.caching` | `RedisCache` |

### api\middleware\compression.py

| Line | Module | Name |
|------|--------|------|
| 6 | `src.middleware.compression` | `CompressionMiddleware` |

### api\middleware\rate_limit.py

| Line | Module | Name |
|------|--------|------|
| 6 | `src.middleware.rate_limit` | `RateLimitMiddleware` |
| 6 | `src.middleware.rate_limit` | `RateLimiter` |

### api\middleware\security.py

| Line | Module | Name |
|------|--------|------|
| 6 | `src.security.security` | `InputSizeLimitMiddleware` |
| 6 | `src.security.security` | `setup_security_middleware` |

### api\validation.py

| Line | Module | Name |
|------|--------|------|
| 7 | `src.core.validation` | `validate_base64_image` |
| 7 | `src.core.validation` | `validate_image_file` |
| 7 | `src.core.validation` | `validate_uuid` |
| 7 | `src.core.validation` | `sanitize_input` |
| 7 | `src.core.validation` | `validate_location_data` |
| 7 | `src.core.validation` | `validate_crop_hint` |
| 7 | `src.core.validation` | `validate_metadata` |
| 7 | `src.core.validation` | `validate_batch_images` |

### src\core\config_manager.py

| Line | Module | Name |
|------|--------|------|
| 10 | `typing` | `List` |

### src\core\configuration_validator.py

| Line | Module | Name |
|------|--------|------|
| 7 | `typing` | `Optional` |
| 9 | `pathlib` | `Path` |

### src\core\validation.py

| Line | Module | Name |
|------|--------|------|
| 8 | `typing` | `Optional` |

### src\dataset\preparation.py

| Line | Module | Name |
|------|--------|------|
| 7 | `os` | `os` |
| 8 | `shutil` | `shutil` |
| 15 | `PIL` | `Image` |
| 16 | `numpy` | `np` |

### src\debugging\monitoring.py

| Line | Module | Name |
|------|--------|------|
| 11 | `typing` | `Optional` |

### src\evaluation\metrics.py

| Line | Module | Name |
|------|--------|------|
| 8 | `sklearn.metrics` | `auc` |
| 16 | `typing` | `Tuple` |

### src\middleware\compression.py

| Line | Module | Name |
|------|--------|------|
| 7 | `typing` | `Callable` |

### src\ood\dynamic_thresholds.py

| Line | Module | Name |
|------|--------|------|
| 9 | `typing` | `List` |

### src\ood\mahalanobis.py

| Line | Module | Name |
|------|--------|------|
| 8 | `numpy` | `np` |

### src\ood\prototypes.py

| Line | Module | Name |
|------|--------|------|
| 8 | `numpy` | `np` |
| 9 | `typing` | `Optional` |
| 12 | `functools` | `lru_cache` |

### src\pipeline\independent_multi_crop_pipeline.py

| Line | Module | Name |
|------|--------|------|
| 10 | `typing` | `Union` |
| 12 | `functools` | `lru_cache` |

### src\router\vlm_pipeline.py

| Line | Module | Name |
|------|--------|------|
| 10 | `typing` | `List` |
| 11 | `pathlib` | `Path` |
| 12 | `numpy` | `np` |

### src\training\phase1_training.py

| Line | Module | Name |
|------|--------|------|
| 40 | `src.ood.prototypes` | `compute_class_prototypes` |

### src\training\phase2_sd_lora.py

| Line | Module | Name |
|------|--------|------|
| 33 | `typing` | `Tuple` |

### src\utils\data_loader.py

| Line | Module | Name |
|------|--------|------|
| 7 | `os` | `os` |
| 17 | `functools` | `lru_cache` |

### src\utils\metrics.py

| Line | Module | Name |
|------|--------|------|
| 1 | `src.evaluation.metrics` | `compute_metrics` |
| 1 | `src.evaluation.metrics` | `compute_protected_retention` |

### src\visualization\visualization.py

| Line | Module | Name |
|------|--------|------|
| 11 | `typing` | `Tuple` |

**Recommendation:** Remove unused imports to improve code clarity, reduce memory footprint, and speed up module loading.

## 4. Inconsistent Import Patterns

The following directories have mixed relative and absolute imports:

- `src\core`
  - `config_manager.py`: relative, absolute
  - `configuration_validator.py`: absolute
  - `model_registry.py`: absolute
  - `pipeline_manager.py`: absolute
  - `schemas.py`: absolute
  - `validation.py`: absolute

**Recommendation:** Standardize import style within each package. Prefer absolute imports for clarity, or use relative imports consistently if the package is designed for it.

## 5. Import Style Analysis

### By File

| File | Import Styles Used |
|------|-------------------|
| `api\database.py` | absolute |
| `api\endpoints\crops.py` | absolute |
| `api\endpoints\diagnose.py` | absolute |
| `api\endpoints\feedback.py` | absolute |
| `api\endpoints\monitoring.py` | absolute |
| `api\graceful_shutdown.py` | absolute |
| `api\main.py` | absolute |
| `api\metrics.py` | absolute |
| `api\middleware\audit.py` | absolute |
| `api\middleware\auth.py` | absolute |
| `api\middleware\caching.py` | absolute |
| `api\middleware\compression.py` | absolute |
| `api\middleware\rate_limit.py` | absolute |
| `api\middleware\security.py` | absolute |
| `api\validation.py` | absolute |
| `src\adapter\independent_crop_adapter.py` | absolute |
| `src\core\config_manager.py` | absolute, relative |
| `src\core\configuration_validator.py` | absolute |
| `src\core\model_registry.py` | absolute |
| `src\core\pipeline_manager.py` | absolute |
| `src\core\schemas.py` | absolute |
| `src\core\validation.py` | absolute |
| `src\dataset\preparation.py` | absolute |
| `src\debugging\monitoring.py` | absolute |
| `src\evaluation\metrics.py` | absolute |
| `src\middleware\audit.py` | absolute |
| `src\middleware\auth.py` | absolute |
| `src\middleware\caching.py` | absolute |
| `src\middleware\compression.py` | absolute |
| `src\middleware\rate_limit.py` | absolute |
| `src\monitoring\metrics.py` | absolute |
| `src\ood\dynamic_thresholds.py` | absolute |
| `src\ood\mahalanobis.py` | absolute |
| `src\ood\prototypes.py` | absolute |
| `src\pipeline\independent_multi_crop_pipeline.py` | absolute |
| `src\router\vlm_pipeline.py` | absolute |
| `src\security\security.py` | absolute |
| `src\training\phase1_training.py` | absolute |
| `src\training\phase2_sd_lora.py` | absolute |
| `src\training\phase3_conec_lora.py` | absolute |
| `src\utils\data_loader.py` | absolute |
| `src\utils\metrics.py` | absolute |
| `src\utils\model_utils.py` | absolute |
| `src\visualization\visualization.py` | absolute |

---

## Appendix: Full Import Graph

### Import Dependencies

**api\database.py** imports:
- logging
- sqlalchemy
- sqlalchemy.orm
- sqlalchemy.pool
- typing

**api\endpoints\crops.py** imports:
- fastapi
- os
- pydantic
- re
- sys
- typing

**api\endpoints\diagnose.py** imports:
- PIL
- base64
- fastapi
- io
- logging
- os
- pathlib
- pydantic
- src.utils.data_loader
- sys
- torch
- typing

**api\endpoints\feedback.py** imports:
- fastapi
- logging
- os
- pydantic
- re
- sys
- typing
- uuid

**api\endpoints\monitoring.py** imports:
- fastapi
- fastapi.responses
- logging
- psutil
- src.monitoring.metrics
- time
- torch
- typing

**api\graceful_shutdown.py** imports:
- api.database
- asyncio
- logging
- signal
- src.middleware.caching
- sys
- typing

**api\main.py** imports:
- asyncio
- fastapi
- fastapi.middleware.cors
- fastapi.responses
- json
- logging
- os
- pathlib
- src.core.config_manager
- src.core.configuration_validator
- sys
- torch
- uvicorn

**api\metrics.py** imports:
- src.monitoring.metrics

**api\middleware\audit.py** imports:
- src.middleware.audit

**api\middleware\auth.py** imports:
- src.middleware.auth

**api\middleware\caching.py** imports:
- src.middleware.caching

**api\middleware\compression.py** imports:
- src.middleware.compression

**api\middleware\rate_limit.py** imports:
- src.middleware.rate_limit

**api\middleware\security.py** imports:
- src.security.security

**api\validation.py** imports:
- src.core.validation

**src\adapter\independent_crop_adapter.py** imports:
- json
- pathlib
- torch
- torch.nn
- typing

**src\core\config_manager.py** imports:
- ....schemas
- ...schemas
- ..configuration_validator
- ..schemas
- .configuration_validator
- .schemas
- json
- logging
- pathlib
- typing

**src\core\configuration_validator.py** imports:
- json
- jsonschema
- logging
- os
- pathlib
- typing

**src\core\model_registry.py** imports:
- asyncio
- dataclasses
- hashlib
- json
- logging
- os
- pathlib
- pickle
- time
- typing

**src\core\pipeline_manager.py** imports:
- asyncio
- dataclasses
- enum
- logging
- time
- typing

**src\core\schemas.py** imports:
- typing

**src\core\validation.py** imports:
- PIL
- base64
- dateutil.parser
- io
- logging
- re
- typing

**src\dataset\preparation.py** imports:
- PIL
- json
- logging
- numpy
- os
- pandas
- pathlib
- random
- shutil
- typing
- zipfile

**src\debugging\monitoring.py** imports:
- datetime
- email.message
- logging
- numpy
- pathlib
- smtplib
- torch
- typing

**src\evaluation\metrics.py** imports:
- logging
- numpy
- sklearn.metrics
- torch
- typing

**src\middleware\audit.py** imports:
- fastapi
- json
- logging
- starlette.middleware.base
- time
- typing
- uuid

**src\middleware\auth.py** imports:
- fastapi
- logging
- starlette.middleware.base

**src\middleware\caching.py** imports:
- asyncio
- fastapi
- hashlib
- json
- logging
- pickle
- redis.asyncio
- starlette.middleware.base
- typing

**src\middleware\compression.py** imports:
- brotli
- fastapi
- gzip
- logging
- starlette.middleware.base
- typing

**src\middleware\rate_limit.py** imports:
- asyncio
- fastapi
- logging
- starlette.middleware.base
- time
- typing

**src\monitoring\metrics.py** imports:
- collections
- dataclasses
- logging
- re
- threading
- time
- typing

**src\ood\dynamic_thresholds.py** imports:
- logging
- numpy
- scipy.stats
- src.utils.model_utils
- torch
- torch.utils.data
- typing

**src\ood\mahalanobis.py** imports:
- numpy
- torch
- typing

**src\ood\prototypes.py** imports:
- argparse
- functools
- logging
- numpy
- src.utils.model_utils
- torch
- torch.utils.data
- typing

**src\pipeline\independent_multi_crop_pipeline.py** imports:
- PIL
- functools
- hashlib
- logging
- numpy
- pathlib
- src.adapter.independent_crop_adapter
- src.router.vlm_pipeline
- src.utils.data_loader
- torch
- typing

**src\router\vlm_pipeline.py** imports:
- builtins
- logging
- numpy
- pathlib
- torch
- typing

**src\security\security.py** imports:
- fastapi
- logging
- starlette.middleware.base
- starlette.middleware.cors

**src\training\phase1_training.py** imports:
- argparse
- logging
- numpy
- pathlib
- peft
- src.ood.prototypes
- src.utils.data_loader
- src.utils.metrics
- src.utils.model_utils
- torch
- torch.nn
- torch.utils.data
- transformers
- typing

**src\training\phase2_sd_lora.py** imports:
- argparse
- logging
- numpy
- pathlib
- peft
- src.utils.data_loader
- src.utils.metrics
- src.utils.model_utils
- torch
- torch.nn
- torch.utils.data
- transformers
- typing

**src\training\phase3_conec_lora.py** imports:
- argparse
- logging
- pathlib
- peft
- src.utils.data_loader
- src.utils.metrics
- src.utils.model_utils
- torch
- torch.nn
- torch.utils.data
- transformers
- typing

**src\utils\data_loader.py** imports:
- PIL
- cv2
- functools
- logging
- numpy
- os
- pathlib
- time
- torch
- torch.utils.data
- torchvision
- typing

**src\utils\metrics.py** imports:
- src.evaluation.metrics

**src\utils\model_utils.py** imports:
- torch
- typing

**src\visualization\visualization.py** imports:
- matplotlib.pyplot
- numpy
- pathlib
- seaborn
- typing

