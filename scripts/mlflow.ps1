param(
    [ValidateSet('setup', 'start', 'stop', 'status', 'import')]
    [string]$Action = 'start',
    [string]$RunId = 'sydney-20261007',
    [ValidateRange(1024, 65535)][int]$Port = 5000
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TrackingRoot = Join-Path $ProjectRoot '.mlflow'
$TrackingPython = Join-Path $ProjectRoot '.venv-mlflow/Scripts/python.exe'
$StatePath = Join-Path $TrackingRoot 'server.json'
$DatabasePath = (Join-Path $TrackingRoot 'tracking.db').Replace('\', '/')
$ArtifactPath = (Join-Path $TrackingRoot 'artifacts').Replace('\', '/')
$ServerUrl = "http://127.0.0.1:$Port"
New-Item -ItemType Directory -Path $TrackingRoot -Force | Out-Null

function Get-OwnedServer {
    if (-not (Test-Path -LiteralPath $StatePath)) { return $null }
    $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($state.pid)"
    if ($process -and $process.CommandLine -match '-m mlflow server' -and
        $process.CommandLine.Contains($DatabasePath)) {
        return @{ Process = $process; State = $state }
    }
    return $null
}

function Stop-OwnedTree([int]$ServerProcessId) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $ServerProcessId" | ForEach-Object {
        Stop-OwnedTree -ServerProcessId $_.ProcessId
    }
    Stop-Process -Id $ServerProcessId -ErrorAction SilentlyContinue
}

Push-Location $ProjectRoot
try {
    if ($Action -eq 'setup') {
        if (-not (Test-Path -LiteralPath $TrackingPython)) {
            python -m venv (Join-Path $ProjectRoot '.venv-mlflow')
            if ($LASTEXITCODE -ne 0) { throw 'Failed to create MLflow environment' }
        }
        & $TrackingPython -m pip install -r requirements-mlflow.lock
        if ($LASTEXITCODE -ne 0) { throw 'Failed to install MLflow dependencies' }
        Write-Host 'Tracking environment ready. Training dependencies were not changed.'
        return
    }
    if ($Action -eq 'stop') {
        $owned = Get-OwnedServer
        if ($owned) {
            Stop-OwnedTree -ServerProcessId $owned.Process.ProcessId
            Remove-Item -LiteralPath $StatePath
            Write-Host 'MLflow stopped. Database and artifacts retained.'
        } else { Write-Host 'No running server owned by this project.' }
        return
    }
    if ($Action -eq 'status') {
        $owned = Get-OwnedServer
        if ($owned) {
            $health = Invoke-WebRequest "$($owned.State.url)/health" -UseBasicParsing -TimeoutSec 5
            Write-Host "MLflow running: $($owned.State.url) (PID $($owned.Process.ProcessId), HTTP $($health.StatusCode))"
        } else { Write-Host 'MLflow is stopped.' }
        return
    }
    if (-not (Test-Path -LiteralPath $TrackingPython)) { throw 'Run ./scripts/mlflow.ps1 setup first.' }
    if ($Action -eq 'import') {
        if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') { throw 'Invalid run ID' }
        $log = Join-Path $TrackingRoot ("import-" + (Get-Date -Format 'yyyyMMddTHHmmss') + '.log')
        & $TrackingPython -m sydney_tracking --run-dir (Join-Path $ProjectRoot "runs/$RunId") --tracking-uri $ServerUrl 2>&1 |
            Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "Import failed; see $log" }
        return
    }
    $owned = Get-OwnedServer
    if ($owned) {
        Write-Host "MLflow already running: $($owned.State.url)"
        return
    }
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is already in use. Choose another -Port; no existing process was stopped."
    }
    $arguments = @('-m', 'mlflow', 'server', '--host', '127.0.0.1', '--port', "$Port", '--workers', '1',
        '--backend-store-uri', "sqlite:///$DatabasePath", '--serve-artifacts', '--artifacts-destination', $ArtifactPath)
    # Start-Process joins arguments into a Windows command line; explicitly quote paths.
    $quotedArguments = $arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
    }
    $process = Start-Process -FilePath $TrackingPython -ArgumentList $quotedArguments -WorkingDirectory $TrackingRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $TrackingRoot 'server.stdout.log') `
        -RedirectStandardError (Join-Path $TrackingRoot 'server.stderr.log')
    @{ pid = $process.Id; url = $ServerUrl; started = (Get-Date).ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding utf8
    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        $process.Refresh()
        if ($process.HasExited) { throw "MLflow exited; see $TrackingRoot/server.stderr.log" }
        try {
            $response = Invoke-WebRequest "$ServerUrl/health" -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -eq 200) {
                Write-Host "MLflow ready: $ServerUrl"
                Write-Host "Persistent database, artifacts, and server logs: $TrackingRoot"
                return
            }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    throw "Server did not become healthy; inspect $TrackingRoot/server.stderr.log"
} finally {
    Pop-Location
}
