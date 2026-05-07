import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def main():
    print("Loading data...")

    df = pd.read_csv('RedShift_Using_Photometric\SDSS_DR18.csv')
    
    print("Filtering and preprocessing data...")

    df_galaxy = df[df['class'] == 'GALAXY'].copy()
    
   
    df_galaxy['u-g'] = df_galaxy['u'] - df_galaxy['g']
    df_galaxy['g-r'] = df_galaxy['g'] - df_galaxy['r']
    df_galaxy['r-i'] = df_galaxy['r'] - df_galaxy['i']
    df_galaxy['i-z'] = df_galaxy['i'] - df_galaxy['z']
    
    base_features = [
        'u', 'g', 'r', 'i', 'z',
        'u-g', 'g-r', 'r-i', 'i-z',
        'petroRad_u', 'petroRad_g', 'petroRad_i', 'petroRad_r', 'petroRad_z',
        'petroFlux_u', 'petroFlux_g', 'petroFlux_i', 'petroFlux_r', 'petroFlux_z',
        'petroR50_u', 'petroR50_g', 'petroR50_i', 'petroR50_r', 'petroR50_z',
        'psfMag_u', 'psfMag_r', 'psfMag_g', 'psfMag_i', 'psfMag_z',
        'expAB_u', 'expAB_g', 'expAB_r', 'expAB_i', 'expAB_z'
    ]
    
    target = 'redshift'
    
    data = df_galaxy[base_features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    
    X = data[base_features]
    y = data[target]
    
    print(f"Dataset shape after preprocessing: {X.shape}")
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Setting up Hyperparameter Tuning (RandomizedSearchCV)...")
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [4, 6, 8, 10],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=15,  
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=2,
        random_state=15,
        n_jobs=-1
    )
    
    print("Training models (this may take a few minutes)...")
    random_search.fit(X_train_scaled, y_train)
    
    print("\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    model = random_search.best_estimator_
    
    print("\nEvaluating the best model...")
    y_pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("----- Evaluation Metrics -----")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")
    print("------------------------------")
    
    print("Generating updated plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=5, color='b')
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Redshift')
    plt.ylabel('Predicted Redshift')
    plt.title('Tuned Galaxy Redshift: True vs. Predicted')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    plot_filename = 'redshift_predictions_tuned.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to '{plot_filename}'")
    plt.close()

if __name__ == "__main__":
    main()
