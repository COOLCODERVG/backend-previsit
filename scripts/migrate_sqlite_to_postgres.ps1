param(
  [string]$SqlitePath = ".\\db.sqlite3",
  [string]$CoreDbUrl = "postgresql://postgres:postgres@localhost:5432/neuravia_core",
  [string]$VectorsDbUrl = "",
  [string]$DocsDbUrl = ""
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
python manage.py migrate --database default

# Dump only app data (avoid Django auth tables which may not exist in the sqlite file).
python manage.py dumpdata api medications `
  --indent 2 `
  --output $dumpFile

if (!(Test-Path $dumpFile)) {
  throw "Dump failed: $dumpFile not created."
}

Write-Host "Dump written to $dumpFile"

# 2) Point Django to Postgres and run migrations
Write-Host "Running migrations on Postgres (core/vectors/documents)..."
$env:CORE_DATABASE_URL = $CoreDbUrl
$VectorsDbUrl = $VectorsDbUrl.Trim()
$DocsDbUrl = $DocsDbUrl.Trim()

if ($VectorsDbUrl -ne "") {
  $env:VECTORS_DATABASE_URL = $VectorsDbUrl
} else {
  $env:VECTORS_DATABASE_URL = $CoreDbUrl
}

if ($DocsDbUrl -ne "") {
  $env:DOCS_DATABASE_URL = $DocsDbUrl
} else {
  $env:DOCS_DATABASE_URL = $CoreDbUrl
}
$env:DJANGO_SETTINGS_MODULE = "previsit.settings"

python manage.py migrate --database default
python manage.py migrate --database vectors
python manage.py migrate --database documents

# 3) Load the dumped data into the core DB
Write-Host "Importing dumped data into Postgres core DB..."
$dumpPath = (Resolve-Path $dumpFile).Path
python manage.py loaddata $dumpPath

Write-Host "Migration complete."

