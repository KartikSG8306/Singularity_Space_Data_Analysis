import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])

    return img, tf.cast(label, tf.float32)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_path = os.path.join(base_dir, "labels.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    df['image_path'] = df['image'].apply(lambda x: os.path.join(base_dir, 'images', x))
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    batch_size = 32
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_df['image_path'].values, train_df['redshift'].values))
    train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_df['image_path'].values, val_df['redshift'].values))
    val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_df['image_path'].values, test_df['redshift'].values))
    test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    
    print("Building EfficientNetB0 model...")

    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(128, 128, 3)
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dense(128, activation='relu', name='dense_features')(x)
    x = layers.Dropout(0.2)(x)
    predictions = layers.Dense(1, name='regressor')(x)
    
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    model.summary()
    
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'notebooks'), exist_ok=True)
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(os.path.join(models_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True)
    ]
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks_list
    )
    
    print("Evaluating model on validation set...")
    val_loss, val_mae, val_rmse = model.evaluate(val_ds)

    print("Evaluating model on untouched test set...")
    test_loss, test_mae, test_rmse = model.evaluate(test_ds)
    
    y_true = test_df['redshift'].values

    y_pred = model.predict(test_ds).flatten()
    
    r2 = r2_score(y_true, y_pred)
    
    print("-" * 30)
    print("Validation Metrics:")
    print(f"MSE  : {val_loss:.4f}")
    print(f"MAE  : {val_mae:.4f}")
    print(f"RMSE : {val_rmse:.4f}")
    print("-" * 30)
    print("Final Test Metrics:")
    print(f"MSE  : {test_loss:.4f}")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print("-" * 30)
    
    metrics_path = os.path.join(models_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("Validation Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"MSE  : {val_loss:.4f}\n")
        f.write(f"MAE  : {val_mae:.4f}\n")
        f.write(f"RMSE : {val_rmse:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("Final Test Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"MSE  : {test_loss:.4f}\n")
        f.write(f"MAE  : {test_mae:.4f}\n")
        f.write(f"RMSE : {test_rmse:.4f}\n")
        f.write(f"R2   : {r2:.4f}\n")
        f.write("-" * 30 + "\n")
    print(f"Metrics saved to {metrics_path}")
    
    weights_path = os.path.join(models_dir, 'model_weights.weights.h5')
    model.save_weights(weights_path)
    print(f"Model weights saved to {weights_path}")
    
    results_df = pd.DataFrame({
        'image': test_df['image'].values,
        'true_redshift': y_true,
        'predicted_redshift': y_pred
    })
    preds_path = os.path.join(models_dir, 'test_predictions.csv')
    results_df.to_csv(preds_path, index=False)
    print(f"Predictions and labels saved to {preds_path}")
    
    print("Extracting features from the 'dense_features' layer...")
    feature_extractor = models.Model(inputs=model.input, outputs=model.get_layer('dense_features').output)
    features = feature_extractor.predict(test_ds)
    features_path = os.path.join(models_dir, 'test_features.npy')
    np.save(features_path, features)
    print(f"Test features (shape: {features.shape}) saved to {features_path}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train MSE/Loss')
    plt.plot(history.history['val_loss'], label='Val MSE/Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.savefig(os.path.join(plots_dir, 'training_curves.png'))
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='b')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True Redshift')
    plt.ylabel('Predicted Redshift')
    plt.title(f'Test True vs Predicted Redshift (R2 = {r2:.3f})')
    plt.grid(True)
    
    plt.savefig(os.path.join(plots_dir, 'redshift_predictions.png'))
    plt.close()
    
    print("Plots saved in plots/ directory.")



if __name__ == "__main__":
    main()
