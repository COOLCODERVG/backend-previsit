param(
  [string]$SqlitePath = ".\\db.sqlite3",
  [string]$CoreDbUrl = "postgresql://postgres:postgres@localhost:5432/neuravia_core"
)

$ErrorActionPreference = "Stop"

Write-Host "Exporting data from SQLite: $SqlitePath"

# 1) Dump core data from SQLite (force SQLite settings)
$env:DJANGO_SETTINGS_MODULE = "previsit.settings_force_sqlite"

$dumpFile = ".\\sqlite_dump.json"

if (!(Test-Path $SqlitePath)) {
  throw "SQLite DB not found at $SqlitePath"
}

# Ensure SQLite schema is up to date with current models before dumping.
Write-Host "Running migrations on SQLite (to align schema for export)..."
python manage.py migrate

# Dump all app data — vectors/documents now live as tables in the same database as everything else.
python manage.py dumpdata api medications vectors documents `
  --indent 2 `
  --output $dumpFile

if (!(Test-Path $dumpFile)) {
  throw "Dump failed: $dumpFile not created."
}

Write-Host "Dump written to $dumpFile"

# 2) Point Django to the single unified Postgres database and run migrations
Write-Host "Running migrations on Postgres..."
$env:CORE_DATABASE_URL = $CoreDbUrl
$env:DJANGO_SETTINGS_MODULE = "previsit.settings"

python manage.py migrate

# 3) Load the dumped data into the unified database
Write-Host "Importing dumped data into Postgres..."
$dumpPath = (Resolve-Path $dumpFile).Path
python manage.py loaddata $dumpPath

Write-Host "Migration complete."

