@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m unittest discover -s tests -p "test_*.py" -v
pause
