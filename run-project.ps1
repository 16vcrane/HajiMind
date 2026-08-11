[CmdletBinding()]
param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BackendUrl = "http://127.0.0.1:8000"
$StdoutLog = Join-Path $env:TEMP "hajimind-uvicorn-stdout.log"
$StderrLog = Join-Path $env:TEMP "hajimind-uvicorn-stderr.log"

function Write-Status {
    param([string]$Message)
    Write-Host "[HajiMind] $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$Description,
        [scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code: $LASTEXITCODE)."
    }
}

function Test-DockerEngine {
    & docker version --format "{{.Server.Version}}" *> $null
    return $LASTEXITCODE -eq 0
}

function Wait-ContainerState {
    param(
        [string]$Container,
        [string]$ExpectedState,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $state = (& docker inspect -f "{{.State.Status}}" $Container 2>$null).Trim()
        if ($state -eq $ExpectedState) {
            return
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    throw "Container '$Container' did not reach '$ExpectedState' within $TimeoutSeconds seconds."
}

function Wait-ContainerHealth {
    param(
        [string]$Container,
        [int]$TimeoutSeconds = 180
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $health = (& docker inspect -f "{{.State.Health.Status}}" $Container 2>$null).Trim()
        if ($health -eq "healthy") {
            return
        }
        Start-Sleep -Seconds 3
    } while ((Get-Date) -lt $deadline)

    throw "Container '$Container' did not become healthy within $TimeoutSeconds seconds."
}

function Test-HajiMindApi {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$BackendUrl/openapi.json" -TimeoutSec 3
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Assert-RequiredEnvValue {
    param([string]$Name)

    $line = Get-Content -LiteralPath (Join-Path $ProjectRoot ".env") |
        Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } |
        Select-Object -Last 1

    if (-not $line) {
        throw ".env is missing required setting '$Name'."
    }

    $value = ($line -split "=", 2)[1].Trim()
    if ([string]::IsNullOrWhiteSpace($value) -or $value -match "^(your_|replace-|<)") {
        throw ".env setting '$Name' is empty or still a placeholder."
    }
}

Set-Location $ProjectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install Docker Desktop and try again."
}

if (-not (Test-DockerEngine)) {
    $dockerDesktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $dockerDesktop)) {
        throw "Docker Desktop is not running and was not found at '$dockerDesktop'. Start Docker Desktop manually, then retry."
    }

    Write-Status "Starting Docker Desktop..."
    Start-Process -FilePath $dockerDesktop
    $deadline = (Get-Date).AddMinutes(3)
    while (-not (Test-DockerEngine)) {
        if ((Get-Date) -ge $deadline) {
            throw "Docker Desktop did not become ready within 3 minutes."
        }
        Start-Sleep -Seconds 3
    }
}

Write-Status "Starting Docker services..."
Invoke-Checked "Docker Compose startup" { docker compose up -d }

Write-Status "Waiting for PostgreSQL, Redis, and Milvus..."
Wait-ContainerHealth "hajimind-postgres"
Wait-ContainerState "hajimind-redis" "running"
Wait-ContainerHealth "milvus-etcd"
Wait-ContainerHealth "milvus-minio"
Wait-ContainerHealth "milvus-standalone"

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env"))) {
    throw "Missing .env. Create it from the template in README.md before starting the project."
}

foreach ($setting in @("ARK_API_KEY", "MODEL", "BASE_URL", "EMBEDDER")) {
    Assert-RequiredEnvValue $setting
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pyLauncher) {
        throw "Python 3.12+ was not found. Install Python, then run this launcher again."
    }

    Write-Status "Creating Python virtual environment..."
    Invoke-Checked "Virtual environment creation" { & $pyLauncher.Source -3.12 -m venv .venv }
}

Write-Status "Checking Python dependencies..."
& $PythonPath -c "import fastapi, uvicorn, sqlalchemy, psycopg, redis, pymilvus" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Status "Installing Python dependencies..."
    Invoke-Checked "Dependency installation" { & $PythonPath -m pip install -e . }
}

# Force the app process to use the Docker services even if .env still contains Milvus Lite settings.
$env:MILVUS_LITE_PATH = ""
$env:MILVUS_HOST = "127.0.0.1"
$env:MILVUS_PORT = "19530"
$env:DATABASE_URL = "postgresql+psycopg://hajimind:hajimind@127.0.0.1:5432/hajimind"
$env:REDIS_URL = "redis://127.0.0.1:6379/0"

if (Test-HajiMindApi) {
    Write-Status "HajiMind is already running."
}
else {
    $portListener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($portListener) {
        throw "Port 8000 is already occupied by another process. Stop that process or change the application port."
    }

    Remove-Item -LiteralPath $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue
    Write-Status "Starting FastAPI on $BackendUrl ..."
    $process = Start-Process `
        -FilePath $PythonPath `
        -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000", "--reload" `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -WindowStyle Hidden `
        -PassThru

    $deadline = (Get-Date).AddSeconds(45)
    while (-not (Test-HajiMindApi)) {
        if ($process.HasExited) {
            $errorText = Get-Content -LiteralPath $StderrLog -Raw -ErrorAction SilentlyContinue
            throw "FastAPI stopped during startup. See $StderrLog`n$errorText"
        }
        if ((Get-Date) -ge $deadline) {
            throw "FastAPI did not respond within 45 seconds. See $StdoutLog and $StderrLog."
        }
        Start-Sleep -Seconds 1
    }
}

Write-Host ""
Write-Host "HajiMind is ready: $BackendUrl/" -ForegroundColor Green
Write-Host "API docs:           $BackendUrl/docs" -ForegroundColor Green
Write-Host "Milvus Attu:        http://127.0.0.1:8080/" -ForegroundColor Green
if (-not $NoBrowser) {
    Start-Process "$BackendUrl/"
}
