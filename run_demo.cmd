@echo off
REM FractureFlow one-click demo (Windows).
REM The two environment variables below are REQUIRED on Windows GBK consoles:
REM   PYTHONUTF8=1            -> force UTF-8 stdio (CJK report output would otherwise crash)
REM   KMP_DUPLICATE_LIB_OK=TRUE -> silence a benign OpenMP duplicate-runtime warning
setlocal
set "PYTHONUTF8=1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
cd /d "%~dp0"
python scripts\demo_run.py %*
endlocal
