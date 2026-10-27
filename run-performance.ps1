param([int]$Port = 7860, [switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Missing .venv. Follow README-local.md to install dependencies.'
}
$env:KOKORO_REQUIRE_CUDA = '1'
$env:KOKORO_CPU_THREADS = '2'
$env:KOKORO_CUDNN_BENCHMARK = '1'
$env:KOKORO_PORT = "$Port"
$env:KOKORO_OPEN_BROWSER = if ($OpenBrowser) { '1' } else { '0' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$process = Get-Process -Id $PID
$previousPriority = $process.PriorityClass
try {
    $process.PriorityClass = 'AboveNormal'
    & $pythonPath 'demo\app.py'
    exit $LASTEXITCODE
}
finally {
    $process.PriorityClass = $previousPriority
}
