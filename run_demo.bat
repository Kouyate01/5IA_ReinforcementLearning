@echo off
REM Batch helper to run the GridWorld demo (uses py or python)

set EXE=
for %%P in (py.exe python.exe) do if not defined EXE (
    where %%P >nul 2>&1 && set EXE=%%P
)

if not defined EXE (
    echo Python interpreter not found. Please install Python or add it to PATH.
    exit /b 1
)

echo Utilisation de l'interpreteur : %EXE%
"%EXE%" demo_gridworld.py
if exist tests\test_gridworld.py (
    echo.
    echo Running tests...
    "%EXE%" tests\test_gridworld.py
)
