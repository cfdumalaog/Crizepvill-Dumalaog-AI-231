@echo off
setlocal EnableExtensions

rem ME2 browser dataset recorder (loopback port 7862).
rem If a recorder already serves the page, open it instead of failing.

for %%I in ("%~dp0..") do set "PROJECT=%%~fI"
for %%I in ("%PROJECT%\..\..") do set "WORKSPACE=%%~fI"
set "PYTHON=%WORKSPACE%\.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:7862/"

if not exist "%PYTHON%" (
  echo Shared Python environment not found: "%PYTHON%"
  pause
  exit /b 1
)

rem --- Already running: open the existing page and exit -------------------
curl.exe -s -m 3 "%URL%" 2>nul | findstr /i /c:"Gradio" >nul
if not errorlevel 1 (
  echo A recorder is already running at %URL% - opening the existing page.
  start "" "%URL%"
  exit /b 0
)

rem --- Fresh start: server gets its own console, then wait for HTTP 200 ---
echo Starting ME2 browser recorder on %URL% ...
pushd "%PROJECT%"
start "ME2 Recorder (port 7862)" "%PYTHON%" "%PROJECT%\record_dataset.py"
popd

set "TRIES=0"
:wait_loop
set /a TRIES+=1
if %TRIES% GTR 60 goto launch_failed
curl.exe -s -m 2 "%URL%" 2>nul | findstr /i /c:"Gradio" >nul
if not errorlevel 1 goto launch_ok
timeout /t 1 /nobreak >nul 2>nul
goto wait_loop

:launch_ok
echo Recorder is up at %URL% - opening the browser.
start "" "%URL%"
exit /b 0

:launch_failed
echo.
echo The recorder did not come up within 60 seconds.
echo If another program owns port 7862, stop it and run this launcher again.
echo Check the "ME2 Recorder (port 7862)" console window for the error.
pause
exit /b 1
