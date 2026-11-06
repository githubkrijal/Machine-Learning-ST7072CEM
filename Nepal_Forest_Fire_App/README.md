# Nepal Forest Fire App — VS Code edition

This folder contains all application source code, four fitted models, prepared input data, evaluation results and the Nepal map outline. The app generates real predictions locally. It needs no weather API keys or new weather downloads. An internet connection is needed for the first dependency installation.

## 1. Open the project

Extract the ZIP first. In VS Code choose **File > Open Folder** and open this **Nepal_Forest_Fire_App** folder. Install the Microsoft Python extension if prompted.

## 2. Create a Python environment (Mac)

Open **Terminal > New Terminal** in VS Code. Use Python 3.10 to match the research environment. Run each line separately:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `python3.10` is unavailable, install Python 3.10 first. The fitted models require scikit-learn 1.7.2, pinned in requirements.txt. Do not substitute a different scikit-learn release.

## 3. Run

```sh
python -m streamlit run application/app.py --server.address 127.0.0.1 --theme.base light --browser.gatherUsageStats false
```

Open **http://localhost:8501** in your browser. Keep the terminal running. Press **Control+C** in the terminal to stop the app.

After setup you can also double-click **Start App.command**. Alternatively select the `.venv` Python interpreter in VS Code and choose **Run and Debug > Run Forest Fire App**.

## Windows setup

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run application/app.py --server.address 127.0.0.1 --theme.base light --browser.gatherUsageStats false
```

Choose the Windows `.venv` interpreter manually in VS Code. If PowerShell blocks activation, activation is optional: use `.venv\Scripts\python.exe` instead of `python` for installation and startup.

## Example for your video

Select **10 March 2016**, **Linear SVM**, and **NP_03_27**. Click **Predict next-day detection**, then **Reveal recorded outcome**. The predicted and recorded outcomes are both detection. Inspect the weather inputs, compare the four models and open Model performance.

The map covers 238 forest-containing grid cells. Predictions cover the historical test period, 1 January–18 June 2016. Weather inputs end on the preceding day. Model scores and daily score percentiles are not calibrated probabilities. No recorded detection does not confirm that no fire occurred. This application demonstrates the historical research and does not provide live warnings.

## Files

- `application/app.py`: Streamlit interface.
- `application/core.py`: loading, feature selection and model inference.
- `application/verify_app.py`: checks scores and decisions against the saved experiment, plus interface smoke tests.
- `research_full/data/`: prepared predictors and feature list.
- `research_full/results/`: four fitted models, thresholds, metrics and saved predictions.
- `data/sources/`: Nepal boundary.
- `.vscode/`: launch configuration and recommended extensions.

All paths are relative to this folder. You can move it outside its original location. No external project folder is required. Only load the supplied trusted model files.

## Verify the installation

```sh
python application/verify_app.py
```

This checks all 40,460 test predictions for each of the four models and the main interface interactions.

## Dataset provenance

The research combines MODIS fire records distributed by ICIMOD (February 2012–June 2016), ERA5 weather accessed through Open-Meteo (2012–2016), a fixed 2011 forest mask, SRTM terrain and a geoBoundaries Nepal boundary. This package includes prepared model inputs rather than the original rasters. It supports inference and demonstration, not rebuilding the original data collection pipeline.

Source services: https://rds.icimod.org/ ; https://open-meteo.com/ ; https://opentopography.org/ ; https://www.geoboundaries.org/ .
