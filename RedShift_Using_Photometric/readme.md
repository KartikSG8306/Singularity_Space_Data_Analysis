# Galaxy Redshift Prediction Using Machine Learning

## Overview

This project predicts galaxy redshift using photometric data from the Sloan Digital Sky Survey (SDSS).

Redshift measures how much a galaxy’s light has shifted toward the red end of the spectrum due to the expansion of the universe. Higher redshift generally indicates greater distance.

The project uses machine learning to estimate redshift from galaxy brightness values across multiple wavelength bands.

---

## Dataset

The dataset contains galaxy observations from SDSS.

### Main Features

| Feature       | Description                              |
| ------------- | ---------------------------------------- |
| u, g, r, i, z | Brightness in different wavelength bands |
| redshift      | Target value to predict                  |

Additional color index features were created:

* u-g
* g-r
* r-i
* i-z

These help capture spectral color differences related to redshift.

---

## Method Used

### 1. Data Preprocessing

* Filtered only GALAXY class objects
* Removed missing and invalid values

### 2. Feature Engineering

Color indices were created using differences between photometric bands.

### 3. Model Training

An XGBoost Regressor was trained to learn the relationship between galaxy photometric properties and redshift.

### 4. Evaluation

The model was evaluated using:

* RMSE
* MAE
* R² Score

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib

---

## Run the Project

Install dependencies:

```bash id="f8x2wq"
pip install pandas numpy scikit-learn xgboost matplotlib
```

Run:

```bash id="x7m3kd"
python redshift_prediction.py
```

---

## Output

The program:

* trains a redshift prediction model,
* evaluates prediction accuracy,
* visualizes true vs predicted redshift values.

---

## Conclusion

This project demonstrates how machine learning and astronomical photometric data can be used to estimate galaxy redshift efficiently without expensive spectroscopic measurements.
