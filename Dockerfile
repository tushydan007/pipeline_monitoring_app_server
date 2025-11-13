# Multi-stage build for optimized production image
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables for build
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_LIBRARY_PATH=/usr/lib/libgdal.so
ENV GEOS_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libgeos_c.so

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    postgresql-client \
    curl \
    libjpeg62-turbo \
    libpng16-16 \
    zlib1g \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_LIBRARY_PATH=/usr/lib/libgdal.so
ENV GEOS_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libgeos_c.so

# Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DJANGO_SETTINGS_MODULE=pipeline_monitoring.settings

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r django && useradd -r -g django django

# Set working directory
WORKDIR /app

# Copy project files
COPY --chown=django:django . .

# Create directories for media and static files with proper permissions
RUN mkdir -p /app/media/satellite/original /app/media/satellite/cog /app/media/pipelines/geojson /app/staticfiles && \
    chown -R django:django /app/media /app/staticfiles && \
    chmod -R 775 /app/media /app/staticfiles

# Create entrypoint script
RUN echo '#!/bin/sh\n\
set -e\n\
\n\
# Wait for database\n\
until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" 2>/dev/null; do\n\
  echo "Waiting for database..."\n\
  sleep 2\n\
done\n\
\n\
echo "Database is ready!"\n\
\n\
# Run migrations\n\
python manage.py migrate --noinput\n\
\n\
# Collect static files\n\
python manage.py collectstatic --noinput --clear\n\
\n\
# Create cache table for database cache (if using db cache)\n\
python manage.py createcachetable 2>/dev/null || true\n\
\n\
# Execute the main command\n\
exec "$@"\n' > /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh && \
    chown django:django /app/docker-entrypoint.sh

# Switch to non-root user
USER django

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

# Use dumb-init for proper signal handling
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/app/docker-entrypoint.sh"]

# Default command
CMD ["gunicorn", "pipeline_monitoring.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]




# # Multi-stage build for optimized production image
# FROM python:3.11-slim AS builder

# # Install build dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     g++ \
#     libgdal-dev \
#     libgeos-dev \
#     libproj-dev \
#     libspatialindex-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Set GDAL environment variables for build
# ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
# ENV C_INCLUDE_PATH=/usr/include/gdal
# ENV GDAL_LIBRARY_PATH=/usr/lib/libgdal.so
# ENV GEOS_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libgeos_c.so

# # Create virtual environment
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # Copy requirements and install Python dependencies
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
#     pip install --no-cache-dir -r requirements.txt

# # Production stage
# FROM python:3.11-slim

# # Install runtime dependencies only
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgdal-dev \
#     libgeos-dev \
#     libproj-dev \
#     libspatialindex-dev \
#     postgresql-client \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Set GDAL environment variables
# ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
# ENV C_INCLUDE_PATH=/usr/include/gdal
# ENV GDAL_LIBRARY_PATH=/usr/lib/libgdal.so
# ENV GEOS_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libgeos_c.so

# # Copy virtual environment from builder
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # Create non-root user for security
# RUN groupadd -r django && useradd -r -g django django

# # Set working directory
# WORKDIR /app

# # Copy project files
# COPY --chown=django:django . .

# # Create directories for media and static files
# RUN mkdir -p /app/media /app/staticfiles && \
#     chown -R django:django /app/media /app/staticfiles

# # Switch to non-root user
# USER django

# # Expose port
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:8000/api/health/ || exit 1

# # Default command (can be overridden in docker-compose)
# CMD ["sh", "-c", "python manage.py migrate && python manage.py collectstatic --noinput && gunicorn pipeline_monitoring.wsgi:application --bind 0.0.0.0:8000 --workers 2 --threads 2 --timeout 120"]


