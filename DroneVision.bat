@echo off
rem DroneVision launcher for Windows 10/11. Double-click to start.
rem The first start prepares everything by itself (Python, packages, helper programs); later starts take seconds.
rem Nothing is installed system-wide and no administrator rights are needed:
rem   bin\uv\  - uv, a single program that downloads Python 3.12 and the packages
rem   .venv\   - the Python environment of DroneVision (safe to delete: it is rebuilt on the next start)
rem Keep this file ASCII-only: cmd.exe reads .bat files in the OEM code page.
setlocal EnableExtensions
title DroneVision
pushd "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "VENV_PY=.venv\Scripts\python.exe"
set "UV_VERSION=0.9.5"

if not exist "dronevision\__main__.py" goto :not_unpacked

call :venv_works
if not errorlevel 1 goto :tools

echo.
echo  DroneVision: first start on this computer.
echo  Preparing Python and its packages. Internet is required, it takes 2-5 minutes.
echo  Nothing is installed outside this folder and the user profile.
echo.
copy /y nul ".dv-write-test" >nul 2>&1
if not exist ".dv-write-test" goto :read_only
del ".dv-write-test" >nul 2>&1

rem 1) uv with its own Python 3.12: the same result on every computer, whatever is installed on it.
call :with_uv
call :venv_works
if not errorlevel 1 goto :tools
rem 2) a Python that is already installed (64-bit, 3.10 or newer).
call :with_system_python
call :venv_works
if not errorlevel 1 goto :tools
goto :no_python

:tools
if not exist "bin\ffmpeg.exe" goto :setup
if not exist "bin\ffprobe.exe" goto :setup
if not exist "bin\dji-log.exe" goto :setup
goto :run

:setup
"%VENV_PY%" -m dronevision setup
if errorlevel 1 goto :fail

:run
echo.
"%VENV_PY%" -m dronevision web --open
if errorlevel 1 goto :fail
popd
exit /b 0

rem --- subroutines -------------------------------------------------------------------------------------------

:venv_works
if not exist "%VENV_PY%" exit /b 1
"%VENV_PY%" -c "import numpy, cv2, pyproj, tkinter" >nul 2>&1
exit /b %errorlevel%

:drop_venv
if exist ".venv" rmdir /s /q ".venv"
goto :eof

:with_uv
set "UV=bin\uv\uv.exe"
if exist "%UV%" goto :uv_ready
where curl >nul 2>&1
if errorlevel 1 goto :eof
where tar >nul 2>&1
if errorlevel 1 goto :eof
echo  [1/3] Downloading uv %UV_VERSION% (20 MB)...
if not exist "bin\uv" mkdir "bin\uv"
curl -fL --retry 3 --connect-timeout 20 -o "bin\uv\uv.zip" "https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
if errorlevel 1 goto :uv_unavailable
tar -xf "bin\uv\uv.zip" -C "bin\uv"
del "bin\uv\uv.zip" >nul 2>&1
if not exist "%UV%" goto :uv_unavailable
:uv_ready
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_LINK_MODE=copy"
call :drop_venv
echo  [2/3] Downloading Python 3.12...
"%UV%" venv --python 3.12 ".venv"
if errorlevel 1 goto :uv_failed
echo  [3/3] Downloading the packages (numpy, opencv, pyproj; about 80 MB)...
"%UV%" pip install --python "%VENV_PY%" --only-binary :all: -r requirements-app.txt
if errorlevel 1 goto :uv_failed
goto :eof
:uv_unavailable
echo  uv could not be downloaded, trying a Python installed on this computer...
goto :eof
:uv_failed
echo  The automatic installation did not finish, trying a Python installed on this computer...
call :drop_venv
goto :eof

:with_system_python
call :find_python
if not defined PY goto :eof
echo  Using Python: %PY%
call :drop_venv
%PY% -m venv ".venv"
if errorlevel 1 goto :eof
rem Ready-made packages only (--only-binary): nothing is compiled on this computer.
"%VENV_PY%" -m pip install --disable-pip-version-check --only-binary=:all: "numpy>=1.26" "opencv-python>=4.9" "pyproj>=3.6"
goto :eof

:find_python
set "PY="
py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10) or sys.maxsize < 2**32)" >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto :eof
rem "python" may be the Microsoft Store stub: it fails this check and is skipped.
python -c "import sys; raise SystemExit(sys.version_info < (3, 10) or sys.maxsize < 2**32)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto :eof
for %%D in ("%LOCALAPPDATA%\Programs\Python" "%ProgramFiles%") do for %%V in (312 313 311 310 314) do call :try_python "%%~D\Python%%V\python.exe"
goto :eof

:try_python
if defined PY goto :eof
if exist "%~1" set PY="%~1"
goto :eof

rem --- messages ----------------------------------------------------------------------------------------------

:not_unpacked
echo.
echo  [!!] The program files are not next to this file.
echo       You probably started DroneVision.bat from inside the ZIP archive.
echo       Unpack the whole archive first (right click - "Extract All..."),
echo       then start DroneVision.bat from the unpacked folder.
goto :hold

:read_only
echo.
echo  [!!] Windows does not allow writing to this folder:
echo       %CD%
echo       Move the DroneVision folder to a simple place, for example C:\DroneVision
echo       (not "Program Files" and not inside the archive), and start this file again.
goto :hold

:no_python
echo.
echo  [!!] The Python environment could not be prepared automatically.
echo       1. Check the internet connection and start this file again.
echo       2. If it fails again: install Python 3.12 (64-bit) from
echo          https://www.python.org/downloads/windows/
echo          keep the default options, tick "Add python.exe to PATH", then start this file again.
echo       3. Still failing: send a screenshot of this window to the developer.
goto :hold

:fail
echo.
echo  [!!] Something went wrong, see the messages above.
echo       Typical reasons: no internet during the first start, antivirus blocking the bin folder.
echo       Start DroneVision-check.bat and send a screenshot of its window to the developer.

:hold
echo.
pause
popd
exit /b 1
