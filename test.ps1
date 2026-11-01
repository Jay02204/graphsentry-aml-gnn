Set-Location $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
