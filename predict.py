"""
predict.py  —  Core prediction function + CLI interface
Task 8: predict_student(weekly_data) → Pass/Fail prediction
"""

import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ── Load artefacts once ────────────────────────────────────────
import os
if not os.path.exists('model.h5') or not os.path.exists('scaler.pkl'):
    raise RuntimeError("model.h5 or scaler.pkl not found. Run train_rnn.py first.")

_model = load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    _scaler = pickle.load(f)

FEATURES  = ['attendance', 'assignment', 'quiz', 'study_hours']
TIMESTEPS = 5
N_FEAT    = len(FEATURES)


def predict_student(weekly_data: list) -> dict:
    """
    Predict Pass/Fail for a student from 5 weeks of data.

    Parameters
    ----------
    weekly_data : list of 5 lists, each with 4 values:
                  [attendance, assignment, quiz, study_hours]
                  Example:
                  [[70,75,80,5],
                   [72,78,82,6],
                   [68,70,79,5],
                   [75,80,85,7],
                   [71,76,83,6]]

    Returns
    -------
    dict with keys: result, label, pass_prob, fail_prob, interpretation
    """
    # Validate input
    if len(weekly_data) != TIMESTEPS:
        raise ValueError(f"Expected 5 weeks of data, got {len(weekly_data)}")
    for i, week in enumerate(weekly_data):
        if len(week) != N_FEAT:
            raise ValueError(f"Week {i+1} needs 4 values [attendance, assignment, quiz, study_hours]")

    # Build (1, 5, 4) array
    X = np.array(weekly_data, dtype=float).reshape(1, TIMESTEPS, N_FEAT)

    # Scale using the same scaler from training
    X_scaled = _scaler.transform(X.reshape(-1, N_FEAT)).reshape(1, TIMESTEPS, N_FEAT)

    # Predict
    pass_prob  = float(_model.predict(X_scaled, verbose=0)[0][0])
    fail_prob  = 1.0 - pass_prob
    prediction = 1 if pass_prob >= 0.5 else 0

    pass_pct = round(pass_prob * 100, 2)
    fail_pct = round(fail_prob * 100, 2)

    if prediction == 1:
        if pass_pct >= 80:
            interp = "🌟 Excellent – very high chance of passing!"
        elif pass_pct >= 65:
            interp = "✅ Good – likely to pass, keep it up!"
        else:
            interp = "⚠️ Borderline pass – consistent effort needed."
    else:
        if fail_pct >= 80:
            interp = "❌ High risk of failure – urgent improvement required."
        elif fail_pct >= 65:
            interp = "⚠️ Likely to fail – focus on weak areas now."
        else:
            interp = "🔶 At risk – small improvements can turn this around."

    return {
        "result"        : prediction,
        "label"         : "Pass" if prediction == 1 else "Fail",
        "pass_prob"     : pass_pct,
        "fail_prob"     : fail_pct,
        "interpretation": interp,
    }


# ── CLI interface ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🎓 RNN Student Performance Evaluator (CLI)")
    print("="*55)
    print("  Enter data for each of the 5 weeks.\n")

    weekly = []
    for w in range(1, 6):
        print(f"  --- Week {w} ---")
        try:
            att  = float(input("    Attendance   (0-100): "))
            asgn = float(input("    Assignment   (0-100): "))
            quiz = float(input("    Quiz         (0-100): "))
            sh   = float(input("    Study hours  (0-15) : "))
            weekly.append([att, asgn, quiz, sh])
        except ValueError:
            print("  ❌ Invalid input. Please enter numbers only.")
            exit(1)

    result = predict_student(weekly)

    print("\n" + "-"*55)
    print(f"  Prediction  : {result['label']}")
    print(f"  Pass Prob   : {result['pass_prob']}%")
    print(f"  Fail Prob   : {result['fail_prob']}%")
    print(f"  {result['interpretation']}")
    print("-"*55 + "\n")
