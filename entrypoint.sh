#!/bin/sh
set -eu

# Run Django migrations against the single unified database.
#
# RUN_MIGRATIONS=0 disables this entirely (used by the worker task family).

RUN_MIGRATIONS="${RUN_MIGRATIONS:-1}"

if [ "$RUN_MIGRATIONS" = "1" ]; then
    echo "[entrypoint] Running migrations..."
    python manage.py migrate --noinput || \
        echo "[entrypoint] WARN: migrate failed"
fi

exec "$@"
