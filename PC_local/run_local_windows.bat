@echo off
setlocal

cd /d "%~dp0"

echo.
echo ==========================================
echo  Classes - lancement local Streamlit
echo ==========================================
echo.

set "PYTHON_CMD="

where python >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=python"

if "%PYTHON_CMD%"=="" (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if "%PYTHON_CMD%"=="" (
    echo Python n'est pas disponible sur ce PC.
    echo Tentative d'installation avec winget...
    echo.

    where winget >nul 2>nul
    if errorlevel 1 (
        echo ERREUR: winget n'est pas disponible.
        echo Demandez a l'administrateur d'installer Python 3.12 ou 3.13.
        echo Pendant l'installation, cochez "Add Python to PATH".
        echo.
        pause
        exit /b 1
    )

    winget install --id Python.Python.3.12 --source winget --scope user --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo ERREUR: l'installation de Python avec winget a echoue.
        echo Demandez a l'administrateur d'installer Python 3.12 ou 3.13.
        echo.
        pause
        exit /b 1
    )

    echo.
    echo Python a ete installe.
    echo Fermez cette fenetre puis relancez run_local_windows.bat
    echo pour que le PATH Windows soit mis a jour.
    echo.
    pause
    exit /b 0
)

if not exist "requirements.txt" (
    echo ERREUR: requirements.txt introuvable.
    echo Lancez ce fichier depuis le dossier du projet.
    echo.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERREUR: app.py introuvable.
    echo Lancez ce fichier depuis le dossier du projet.
    echo.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creation de l'environnement virtuel...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo.
        echo ERREUR: impossible de creer le venv.
        pause
        exit /b 1
    )
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo ERREUR: impossible d'activer le venv.
    pause
    exit /b 1
)

echo Installation / verification des dependances...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERREUR: installation des dependances echouee.
    echo Verifiez la connexion Internet ou demandez a l'administrateur reseau.
    pause
    exit /b 1
)

echo.
echo L'application va demarrer.
echo.
echo Acces sur ce PC:
echo   http://localhost:8501
echo.
echo Acces depuis un autre PC du reseau:
echo   utilisez l'adresse Network URL affichee par Streamlit
echo   ou l'IP de ce PC suivie de :8501
echo.

streamlit run app.py --server.address 0.0.0.0

echo.
echo Application arretee.
pause
