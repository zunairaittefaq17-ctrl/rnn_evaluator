"""
train_rnn.py  —  RNN Student Performance Prediction System
Run this locally to train and save model.h5 + scaler.pkl
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models   import Sequential
from tensorflow.keras.layers   import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 60)
print("  RNN Student Performance Prediction System")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")

# ── STEP 1 : Load ──────────────────────────────────────────────
print("\n[1] Loading dataset ...")
df = pd.read_excel("dataset.xlsx")
print(f"    Shape: {df.shape}  |  Students: {df['student_id'].nunique()}")
print(df.head(10).to_string(index=False))

# ── STEP 2 : Build sequences ───────────────────────────────────
print("\n[2] Building sequences (samples, timesteps, features) ...")

FEATURES  = ['attendance', 'assignment', 'quiz', 'study_hours']
TIMESTEPS = 5
N_FEAT    = len(FEATURES)

seqs, labels = [], []
for sid, grp in df.groupby('student_id'):
    grp  = grp.sort_values('week')
    seqs.append(grp[FEATURES].values)
    labels.append(int(grp['result'].iloc[-1]))

X = np.array(seqs)    # (50, 5, 4)
y = np.array(labels)  # (50,)
print(f"    X shape: {X.shape}   y shape: {y.shape}")
print(f"    Pass: {y.sum()}  Fail: {(y==0).sum()}")

# ── STEP 3 : Split ─────────────────────────────────────────────
print("\n[3] Splitting 80/20 ...")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)
print(f"    Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}")

# ── STEP 4 : Scale ─────────────────────────────────────────────
print("\n[4] Scaling features ...")
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr.reshape(-1, N_FEAT)).reshape(X_tr.shape)
X_te_s = scaler.transform(X_te.reshape(-1, N_FEAT)).reshape(X_te.shape)
print("    StandardScaler applied (mean=0, std=1)")

# ── STEP 5 : Build model ───────────────────────────────────────
print("\n[5] Building LSTM model ...")
model = Sequential([
    LSTM(64, input_shape=(TIMESTEPS, N_FEAT), return_sequences=True,  name="LSTM_1"),
    Dropout(0.3, name="Drop_1"),
    LSTM(32, return_sequences=False, name="LSTM_2"),
    Dropout(0.2, name="Drop_2"),
    Dense(16, activation='relu',    name="Dense_1"),
    Dense(1,  activation='sigmoid', name="Output"),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ── STEP 6 : Train ─────────────────────────────────────────────
print("\n[6] Training ...")
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
history = model.fit(X_tr_s, y_tr, epochs=50, batch_size=8,
                    validation_split=0.15, callbacks=[es], verbose=1)
print(f"\n    Done — {len(history.history['loss'])} epochs ran.")

# ── STEP 7 : Evaluate ──────────────────────────────────────────
print("\n[7] Evaluating ...")
y_pred = (model.predict(X_te_s, verbose=0).flatten() >= 0.5).astype(int)
acc = accuracy_score(y_te, y_pred)
cm  = confusion_matrix(y_te, y_pred)
print(f"\n    Accuracy : {acc*100:.2f}%")
print("\n    Confusion Matrix:\n", cm)
print("\n    Classification Report:")
print(classification_report(y_te, y_pred, target_names=['Fail','Pass']))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history.history['accuracy'],     label='Train'); axes[0].plot(history.history['val_accuracy'], label='Val', linestyle='--')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history.history['loss'],         label='Train', color='red'); axes[1].plot(history.history['val_loss'], label='Val', color='purple', linestyle='--')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail','Pass'], yticklabels=['Fail','Pass'], ax=axes[2])
axes[2].set_title('Confusion Matrix'); axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Actual')
plt.tight_layout(); plt.savefig('training_results.png', dpi=150)
print("    Plot saved → training_results.png")

# ── STEP 8 : Save ──────────────────────────────────────────────
print("\n[8] Saving model and scaler ...")
model.save('model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("    model.h5   saved ✅")
print("    scaler.pkl saved ✅")
print("\n" + "="*60)
print("  Training complete! Run: streamlit run app.py")
print("="*60)
