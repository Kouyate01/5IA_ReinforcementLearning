# PowerShell helper to execute the GridWorld demo or tests
# It will prefer the "py" launcher, then "python". Display an error if neither is available.

$exe = (Get-Command py -ErrorAction SilentlyContinue)?.Source
if (-not $exe) {
    $exe = (Get-Command python -ErrorAction SilentlyContinue)?.Source
}

if (-not $exe) {
    Write-Error "Python interpreter not found. Please install Python or add it to your PATH."
    exit 1
}

Write-Host "Utilisation de l'interpréteur : $exe" -ForegroundColor Green

# lancer la démo
& $exe demo_gridworld.py

# optionally run the test script if it exists
if (Test-Path "tests/test_gridworld.py") {
    Write-Host "\nRunning tests..." -ForegroundColor Yellow
    & $exe tests/test_gridworld.py
}
