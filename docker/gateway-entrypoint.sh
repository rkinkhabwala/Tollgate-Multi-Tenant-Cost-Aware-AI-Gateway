#!/bin/sh
# Apply migrations (as the owner role), then serve.
set -e
alembic upgrade head
exec uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --no-access-log --proxy-headers
