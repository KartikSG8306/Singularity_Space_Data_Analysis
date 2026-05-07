# Meteor Detection Using Signal Analysis

## Overview

This project detects meteor events from radio antenna signal data using signal processing techniques.

When a meteor enters Earth’s atmosphere, it creates an ionized trail that reflects radio waves. This causes sudden spikes in the received radio signal. The project identifies these spikes and counts meteor events automatically.

---

## Dataset

The dataset contains:

| Column    | Description                    |
| --------- | ------------------------------ |
| Time      | Timestamp                      |
| Level     | Signal strength (main feature) |
| Noise     | Background noise               |
| Frequency | Operating frequency            |
| Bandwidth | Signal bandwidth               |

Meteor events appear as sharp peaks in the `Level` signal.

---

## Method Used

### 1. Rolling Median Baseline

A rolling median is used to estimate the local background signal while ignoring meteor spikes.

### 2. Signal Extraction

The baseline is subtracted from the original signal to isolate unusual variations.

### 3. Robust Thresholding

Median Absolute Deviation (MAD) is used to estimate normal noise fluctuations and create an adaptive detection threshold.

### 4. Peak Detection

`scipy.signal.find_peaks()` is used to detect meteor peaks based on:

* minimum peak height
* minimum distance between peaks

---

## Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Matplotlib

---

## Run the Project

Install dependencies:

```bash id="g6xv3k"
pip install pandas numpy scipy matplotlib
```

Run:

```bash id="9v7w3m"
python meteor_detection.py
```

---

## Output

The program:

* counts detected meteors,
* prints detection statistics,
* plots signal peaks visually.

---

## Conclusion

This project demonstrates how signal processing and anomaly detection techniques can be used to identify meteor events from real radio antenna data.
