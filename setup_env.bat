@echo off
echo Setting up Native Language Identification environment...
echo ======================================================

echo Creating virtual environment...
python -m venv nli_env

echo Activating virtual environment...
call nli_env\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo Setting up project directories...
python setup.py

echo Testing installation...
python test_installation.py

echo.
echo ======================================================
echo Setup completed!
echo.
echo To activate the environment in the future, run:
echo   nli_env\Scripts\activate
echo.
echo To start working on the project, run:
echo   python main.py --help
echo ======================================================