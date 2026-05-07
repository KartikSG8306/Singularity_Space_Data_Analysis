# Galaxy Redshift Estimation

This project predicts galaxy redshift (`z`) using image-based data from SDSS galaxy cutouts. It was built as a club recruitment project to explore how visual galaxy features can help estimate distance.

## Approach

- Used `SDSS_DR18.csv` as the source dataset.
- Filtered only objects classified as `GALAXY`.
- Downloaded 2000 SDSS image cutouts using RA and DEC values.
- Trained an EfficientNetB0-based CNN regression model.
- Used train, validation, and test splits for evaluation.

## Model

The model uses:

- EfficientNetB0 pretrained backbone
- Global average pooling
- Dense regression head
- MSE loss for redshift prediction

## Results

Final test performance:

```text
MSE  : 0.0012
MAE  : 0.0259
RMSE : 0.0341
R2   : 0.3077
```

The positive R2 score shows that the model learned useful information from the galaxy images.

## How to Run

```powershell
python train.py
```

Main outputs are saved in:

- `models/`
- `plots/`

## Files

```text
download_images.py   # downloads galaxy images
train.py             # trains and evaluates the model
labels.csv           # image names and redshift labels
images/              # downloaded galaxy cutouts
models/              # saved model and metrics
plots/               # result plots
```

## Note

This project uses SDSS RGB image cutouts. A more advanced version could use true multi-band FITS data from the `u`, `g`, `r`, `i`, and `z` bands for better redshift estimation.
