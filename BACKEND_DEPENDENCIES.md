# Backend Dependencies List

This document lists all Python packages and dependencies used in the backend Django application.

## Core Framework Dependencies

### Django & REST Framework

- **Django** (>=5.2.7) - Web framework
- **djangorestframework** (>=3.15.0) - Django REST Framework for API
- **djangorestframework-simplejwt** (>=5.3.1) - JWT authentication for DRF
- **djoser** (>=2.2.0) - REST implementation of Django authentication

### Database

- **psycopg2-binary** (>=2.9.9) - PostgreSQL adapter for Python

### CORS & Middleware

- **django-cors-headers** (>=4.5.0) - Django app for handling CORS headers

## Geospatial & Image Processing

### GDAL & Geospatial Libraries

- **GDAL** (>=3.8.0) - Geospatial Data Abstraction Library
- **rasterio** (>=1.3.9) - Raster I/O library for Python
- **geopandas** (>=0.14.3) - Working with geospatial data in Python
- **Shapely** (>=2.0.2) - Manipulation and analysis of geometric objects
- **Fiona** (>=1.9.6) - Reading and writing spatial data files

### Image Processing

- **Pillow** (>=11.0.0) - Python Imaging Library (PIL fork)
- **numpy** (>=2.1.0) - Numerical computing library
- **scikit-image** (>=0.23.0) - Image processing algorithms
- **opencv-python-headless** (>=4.10.0) - Computer vision library (headless version)
- **pandas** (>=2.2.1) - Data analysis and manipulation library

### Scientific Computing

- **scipy** - Scientific computing library (dependency of scikit-image)

## Task Queue & Caching

### Celery & Redis

- **celery** (>=5.3.6) - Distributed task queue
- **redis** (>=5.0.4) - Redis Python client
- **django-celery-beat** (>=2.7.0) - Database-backed Periodic Tasks for Celery

## Utilities

### Configuration & Environment

- **python-dotenv** (>=1.0.1) - Load environment variables from .env file

## Usage in Code

### Directly Imported Packages

**Django Core:**

- `django` - Core Django framework
- `django.contrib.auth` - Authentication system
- `django.contrib.admin` - Admin interface
- `django.db` - Database models
- `django.core.mail` - Email functionality
- `django.http` - HTTP utilities
- `django.utils` - Utility functions

**REST Framework:**

- `rest_framework` - Django REST Framework
- `rest_framework_simplejwt` - JWT authentication
- `rest_framework.viewsets` - ViewSets
- `rest_framework.decorators` - Decorators
- `rest_framework.response` - Response classes

**Djoser:**

- `djoser.views` - User management views
- `djoser.serializers` - User serializers
- `djoser.signals` - Authentication signals

**Geospatial:**

- `rasterio` - GeoTIFF processing
- `numpy` - Array operations
- `skimage` (scikit-image) - Image analysis (filters, feature, morphology, segmentation)
- `scipy.ndimage` - Image processing

**Celery:**

- `celery` - Task queue
- `celery.shared_task` - Shared task decorator

**CORS:**

- `corsheaders` - CORS middleware

## Complete Dependency List

```
Django>=5.2.7
djangorestframework>=3.15.0
djangorestframework-simplejwt>=5.3.1
djoser>=2.2.0
django-cors-headers>=4.5.0
psycopg2-binary>=2.9.9
Pillow>=11.0.0
GDAL>=3.8.0
rasterio>=1.3.9
numpy>=2.1.0
celery>=5.3.6
redis>=5.0.4
django-celery-beat>=2.7.0
geopandas>=0.14.3
Shapely>=2.0.2
Fiona>=1.9.6
scikit-image>=0.23.0
opencv-python-headless>=4.10.0
pandas>=2.2.1
python-dotenv>=1.0.1
scipy  # Dependency of scikit-image
```

## Notes

1. **GDAL** on Windows requires Anaconda installation with proper paths configured
2. **PostgreSQL** database is required (use psycopg2-binary for easier installation)
3. **Redis** server is required for Celery task queue
4. Some packages (like geopandas, rasterio, fiona) have system-level dependencies (GDAL, GEOS, PROJ) that need to be installed separately
5. For Windows development, GDAL paths are configured in `settings.py` to use Anaconda environment
6. For Docker/Linux, GDAL libraries are installed in the Dockerfile

## Optional Development Dependencies

These are listed in Pipfile but not strictly required for production:

- pytest
- pytest-django
- black (code formatting)
- flake8 (linting)
