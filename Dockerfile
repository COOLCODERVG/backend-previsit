FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_SETTINGS_MODULE=previsit.settings \
    # SSL defaults so the API verifies AWS RDS certs out of the box.
    # Override via Secrets Manager / .env if needed (e.g. DB_SSLMODE=disable for local docker-compose).
    DB_SSLMODE=verify-full \
    DB_SSLROOTCERT=/etc/ssl/rds/global-bundle.pem

# WeasyPrint runtime deps (PDF rendering) + libpq for psycopg
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq5 \
        libcairo2 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
        fonts-dejavu-core \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# AWS RDS global trust bundle for sslmode=verify-full.
# Pinned location is referenced by DB_SSLROOTCERT above and by previsit/settings.py.
RUN mkdir -p /etc/ssl/rds \
    && curl -fsSL -o /etc/ssl/rds/global-bundle.pem \
        https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem \
    && chmod 0644 /etc/ssl/rds/global-bundle.pem

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Drop privileges
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000

COPY --chown=app:app entrypoint.sh /usr/local/bin/entrypoint.sh
# Defense-in-depth against CRLF line endings sneaking in from a Windows host
# checkout. Without this `set -eu\r` fails as: "Illegal option -".
USER root
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod 0755 /usr/local/bin/entrypoint.sh
USER app
ENTRYPOINT ["/bin/sh", "/usr/local/bin/entrypoint.sh"]

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
