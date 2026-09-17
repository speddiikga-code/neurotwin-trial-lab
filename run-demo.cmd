@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 start.py
) else (
  if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
    "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" start.py
  ) else (
    python start.py
  )
)
if errorlevel 1 (
  echo Run failed. Install Python 3.12 from python.org, then try again.
) else (
  start "" "results\report.html"
)
pause
