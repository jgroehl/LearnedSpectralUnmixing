@echo off
REM This script was copletely generated by ChatGPT...

REM Set the path to the virtual environment
set "venv_path=%~dp0LearnedSpectralUnmixing\Scripts"

REM List of datasets and wavelengths
set "datasets=ACOUS BG_0-100 BG_60-80 BG_H2O HET_0-100 HET_60-80 ILLUM_5mm ILLUM_POINT INVIS INVIS_ACOUS INVIS_SKIN INVIS_SKIN_ACOUS MSOT MSOT_ACOUS MSOT_ACOUS_SKIN MSOT_SKIN RES_0.15 RES_0.15_SMALL RES_0.6 RES_1.2 SKIN SMALL WATER_2cm WATER_4cm"
set "wavelengths=3, 5, 10, 15, 20, 41"

REM Loop through datasets and wavelengths
for %%d in (%datasets%) do (
  for %%w in (%wavelengths%) do (
    echo Running with Dataset: %%d, Wavelength: %%w

    REM Activate the virtual environment
    call "%venv_path%\activate.bat"

    REM Run the Python script
    python train_LSTM_models.py %%d %%w

    REM Deactivate the virtual environment
    call "%venv_path%\deactivate.bat"
  )
)