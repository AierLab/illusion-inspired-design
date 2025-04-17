@echo off
REM Set up timestamp-based log directory
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set timestamp=%%i
set log_dir=log\%timestamp%
mkdir %log_dir%

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Define config names
set config_1=m_A-none-indl224
set config_2=m_B-none-imagenet1k
set config_3=m_X-103-imagenet1k

REM Run training for each config
@REM call :run_config %config_1%
@REM call :run_config %config_2%
call :run_config %config_3%

goto :eof

:run_config
set config_name=%1
echo Running config: %config_name%
python train.py --config_name "%config_name%" > "%log_dir%\%config_name%.log" 2>&1
echo Done: %config_name%
echo Log: %log_dir%\%config_name%.log
echo ======================================================================
goto :eof
