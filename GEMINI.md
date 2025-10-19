# GEMINI.md

**Speak in Japanese!!**

## Project Overview

This project is for the Kaggle competition "House Prices: Advanced Regression Techniques". The goal is to predict the sales price of residential homes in Ames, Iowa, based on 79 explanatory variables describing various aspects of the homes.

This is a regression problem. The evaluation metric for this competition is the Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.

## Getting Started

### 1. Download the Data

This project uses the official Kaggle CLI to download and manage the dataset.

**Prerequisites:**
1.  **Install Kaggle CLI:** If you haven't already, install the library:
    ```bash
    pip install kaggle
    ```
2.  **API Credentials:** You need to have your Kaggle API credentials (`kaggle.json`) set up. You can get this from your Kaggle account page under the "API" section.
3.  **Accept Competition Rules:** Before downloading, you must accept the competition rules on the Kaggle website:
    [https://www.kaggle.com/c/house-prices-advanced-regression-techniques/rules](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/rules)

**Download Command:**
Once the prerequisites are met, run the following command from the project root to download the data into the `data/` directory:
```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data
```
This will download a zip file, which should be extracted into the `data` directory.

### 2. Project Structure

A suggested project structure is as follows:

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_feature_engineering.py
│   └── 03_model_training.py
├── src/
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       └── visualize.py
├── requirements.txt
└── GEMINI.md
```

## Building and Running

*TODO: Add instructions on how to build and run the project once the code is developed.*

## Development Conventions

*   **Code Style:** Follow PEP 8 for Python code.
*   **Notebooks:** This project uses [Marimo](https://marimo.io/) for interactive notebooks. All notebooks are saved as standard Python (`.py`) files in the `notebooks` directory.
*   **Source Code:** Place reusable code in the `src` directory.
*   **Data:** Keep raw data in the `data` directory.
*   **Dependencies:** List all Python dependencies in the `requirements.txt` file.
