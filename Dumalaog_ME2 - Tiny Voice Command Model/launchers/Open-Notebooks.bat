@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT=%%~fI"
for %%I in ("%PROJECT%\..\..") do set "WORKSPACE=%%~fI"
set "PYTHON=%WORKSPACE%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Shared Python environment not found: "%PYTHON%"
  pause
  exit /b 1
)
pushd "%PROJECT%"
"%PYTHON%" "%PROJECT%\scripts\start_jupyter.py"
set "RESULT=%ERRORLEVEL%"
popd
if not "%RESULT%"=="0" pause
exit /b %RESULT%
