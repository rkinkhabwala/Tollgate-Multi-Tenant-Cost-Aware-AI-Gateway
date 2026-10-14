#!/bin/sh
# Apply migrations (as the owner role), then serve. The access log carries paths
# only (never bodies or headers) and goes through the JSON + scrubbing formatter.
set -e
alembic upgrade head
exec uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --proxy-headers
