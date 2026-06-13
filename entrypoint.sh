#!/bin/sh
set -eu

# Run Django migrations against each configured database.
#
# The Django settings module aliases the `vectors` and `documents` DB entries
# to the `default` connection when the dedicated env var isn't set. That means
# the *Django alias* always exists, even on local SQLite, so we should always
# run `migrate --database <alias>` and let Django decide whether the target is
# a real Postgres instance or a no-op alias.
#
# RUN_MIGRATIONS=0 disables this entirely (used by the worker task family).

RUN_MIGRATIONS="${RUN_MIGRATIONS:-1}"

if [ "$RUN_MIGRATIONS" = "1" ]; then
    echo "[entrypoint] Running migrations on default..."
    python manage.py migrate --database default --noinput || \
        echo "[entrypoint] WARN: default migrate failed"

    echo "[entrypoint] Running migrations on vectors..."
    python manage.py migrate --database vectors --noinput || \
        echo "[entrypoint] WARN: vectors migrate failed"

    echo "[entrypoint] Running migrations on documents..."
    python manage.py migrate --database documents --noinput || \
        echo "[entrypoint] WARN: documents migrate failed"
fi

exec "$@"
