@echo off
rem Temporary runner for mypy to avoid PowerShell parsing issues
scripts\python.cmd -m mypy src --exclude "(^\.venv|\.venv)" --show-error-codes
