# Exoplanet Prediction Hub

## Overview

This project is a web-based platform that provides access to several machine learning models for classifying celestial objects and predicting planetary events. It is designed to be a user-friendly interface for astronomers, researchers, and enthusiasts to get predictions on their data.

## Features

The platform currently includes the following prediction models:

*   **Exoplanet Classification:** Classifies a celestial object as a "Confirmed Exoplanet" or a "False Positive" based on a set of 25 stellar and planetary features.
*   **Microlensing Prediction:** Predicts the probability of a planetary microlensing event from time-series data.
*   **Astrometry Prediction:** Predicts the probability of a star being a planet host based on astrometric features.
*   **Radial Velocity Classification:** Classifies a radial velocity curve to determine if it belongs to a planet host.
*   **Image Classification (Disabled):** A placeholder for a future image classification model that requires a GPU.

## Getting Started

### Prerequisites

*   Python 3.7+
*   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the web application, use the following command from the project's root directory:

```bash
uvicorn app.main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## How to Use

Navigate to the home page to see the available prediction models. Each model has its own page with instructions on how to provide the input data.

### Exoplanet Classification

*   **Manual Input:** Fill in the form with the 25 required features.
*   **CSV Upload:** Upload a CSV file with the features for one or more candidates.

### Microlensing Prediction

*   **File Upload:** Upload a data file (`.npy`, `.csv`, `.txt`, `.tsv`) containing the time-series data.
*   **Dummy Data:** Use the "Predict Dummy Data" button to run a prediction on a pre-defined example.

### Astrometry Prediction

*   **Manual Input:** Fill in the form with the 10 required astrometric features.
*   **Pre-selected Data:** Use the "Load Planet-like Example" and "Load Non-planet Example" buttons to populate the form with example data.
*   **CSV Upload:** Upload a CSV file for batch prediction.

### Radial Velocity Classification

*   **File Upload:** Upload a data file (`.tbl`, `.csv`, `.txt`) containing the radial velocity curve.
*   **Text Input:** Paste the time-series data (time, rv, rv_err) into the text area.
*   **Example Data:** Use the "Load Example Data" button to populate the text area with an example.

## Data Storage

Please note: All data you submit is stored in the background in separate CSV files for each model. This data is collected to help us further train and improve our models.
