"""
TSLA LSTM - Train (v8)
- Predicts returns (not prices) — output is unbounded, no sigmoid
- 17 input features
- Huber loss instead of MSE (more robust to return outliers)
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR  = "data/tsla_lstm_data"
MODEL_DIR = "models"
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}  range: {y_train.min():.4f} → {y_train.max():.4f}")
    print(f"y_test  : {y_test.shape}")
    print(f"Features: {meta['features']}")
    return X_train, X_test, y_train, y_test, meta


def build_model(seq_len, n_features, forecast_len, lr):
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=(seq_len, n_features)
        ),
        Dropout(0.2),

        LSTM(32, return_sequences=False),
        Dropout(0.2),

        Dense(32, activation="relu"),

        # Linear output — returns can be positive or negative
        Dense(forecast_len, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            clipnorm=1.0
        ),
        # Huber loss: like MSE but less sensitive to large return outliers
        # (e.g. -30% COVID crash days won't destroy the gradient)
        loss=tf.keras.losses.Huber(delta=0.1),
        metrics=["mae"]
    )
    return model


def get_callbacks(model_dir):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def plot_history(history, model_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Training History (v8 — Returns)", fontsize=13, fontweight="bold")

    axes[0].plot(history.history["loss"],     label="Train Loss (Huber)")
    axes[0].plot(history.history["val_loss"], label="Val Loss (Huber)")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["mae"],     label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("MAE (in return units, ~0.01 = 1%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(model_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Chart saved: {path}")


def main():
    print("Available devices:")
    for d in tf.config.list_physical_devices():
        print(f"  {d}")
    gpu = tf.config.list_physical_devices("GPU")
    print(f"\n  {'✅ GPU active' if gpu else '❌ No GPU'}\n")

    X_train, X_test, y_train, y_test, meta = load_data(DATA_DIR)
    seq_len      = meta["sequence_len"]
    n_features   = meta["n_features"]
    forecast_len = meta["forecast_len"]

    model = build_model(seq_len, n_features, forecast_len, LEARNING_RATE)
    model.summary()

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(MODEL_DIR),
        verbose=1,
    )

    print("\nEvaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Huber Loss : {test_loss:.6f}")
    print(f"  Test MAE        : {test_mae:.6f}  (~{test_mae*100:.2f}% avg return error)")

    model.save(os.path.join(MODEL_DIR, "best_model.keras"))
    print(f"\n✅ Model saved to: {MODEL_DIR}/best_model.keras")

    plot_history(history, MODEL_DIR)
    print("\nNext → run predict_lstm.py")


if __name__ == "__main__":
    main()
