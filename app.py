"""
app.py  —  Streamlit UI for RNN Student Performance Predictor
Auto-trains the model on first launch if model.h5 is missing.
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="RNN Student Predictor",
    page_icon="🧠",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════
# EMBEDDED DATASET (for cloud auto-training — no Excel needed)
# ══════════════════════════════════════════════════════════════
DATASET = [
    # student_id, week, attendance, assignment, quiz, study_hours, result
    [1,1,59,53,65,4,0],[1,2,60,51,55,5,0],[1,3,56,50,60,3,0],[1,4,60,51,50,5,0],[1,5,65,62,67,4,0],
    [2,1,79,80,79,2,1],[2,2,71,75,83,4,1],[2,3,76,65,67,3,1],[2,4,73,70,68,2,1],[2,5,69,74,78,2,1],
    [3,1,58,50,68,5,0],[3,2,61,61,57,4,0],[3,3,58,64,67,5,0],[3,4,58,59,55,3,0],[3,5,63,62,56,3,0],
    [4,1,73,78,68,2,1],[4,2,74,79,81,3,1],[4,3,77,78,72,1,1],[4,4,70,78,76,1,1],[4,5,68,69,71,4,1],
    [5,1,53,51,59,2,0],[5,2,54,44,44,2,0],[5,3,52,52,49,3,0],[5,4,57,50,52,5,0],[5,5,51,45,60,4,0],
    [6,1,75,67,65,5,1],[6,2,76,66,64,4,1],[6,3,74,79,78,4,1],[6,4,71,70,79,4,1],[6,5,80,81,81,3,1],
    [7,1,57,56,66,5,0],[7,2,59,55,57,2,0],[7,3,60,64,58,4,0],[7,4,60,54,62,2,0],[7,5,67,59,65,5,0],
    [8,1,74,66,60,2,1],[8,2,74,80,73,2,1],[8,3,69,68,67,5,1],[8,4,76,67,63,4,1],[8,5,78,75,78,4,1],
    [9,1,74,70,81,2,1],[9,2,68,82,74,3,1],[9,3,75,65,66,2,1],[9,4,74,81,81,4,1],[9,5,71,67,67,5,1],
    [10,1,69,60,60,3,1],[10,2,65,70,64,4,1],[10,3,69,61,68,4,1],[10,4,62,62,61,4,1],[10,5,62,61,66,3,1],
    [11,1,61,64,61,5,1],[11,2,64,76,60,4,1],[11,3,72,60,76,5,1],[11,4,70,78,67,2,1],[11,5,72,66,77,5,1],
    [12,1,52,56,48,3,0],[12,2,49,53,53,1,0],[12,3,45,53,40,1,0],[12,4,45,40,44,1,0],[12,5,42,53,50,4,0],
    [13,1,55,43,62,4,0],[13,2,49,51,47,3,0],[13,3,57,64,47,5,0],[13,4,59,48,63,5,0],[13,5,53,48,49,5,0],
    [14,1,75,67,70,1,1],[14,2,75,81,85,4,1],[14,3,77,71,67,1,1],[14,4,69,79,75,3,1],[14,5,70,76,68,2,1],
    [15,1,61,58,52,1,0],[15,2,55,64,50,1,0],[15,3,56,52,51,4,0],[15,4,53,64,59,4,0],[15,5,56,52,49,4,0],
    [16,1,59,63,60,1,0],[16,2,53,46,65,3,0],[16,3,60,55,62,2,0],[16,4,62,58,57,3,0],[16,5,65,60,63,2,0],
    [17,1,76,71,72,3,1],[17,2,79,76,81,2,1],[17,3,78,80,79,4,1],[17,4,72,75,74,3,1],[17,5,77,74,76,5,1],
    [18,1,57,52,55,3,0],[18,2,55,54,60,4,0],[18,3,60,55,52,2,0],[18,4,58,48,60,3,0],[18,5,55,50,57,4,0],
    [19,1,71,78,79,2,1],[19,2,75,77,76,4,1],[19,3,73,80,82,3,1],[19,4,76,79,78,4,1],[19,5,70,76,75,5,1],
    [20,1,60,57,62,4,0],[20,2,55,56,58,3,0],[20,3,62,60,65,5,0],[20,4,57,55,61,4,0],[20,5,60,60,64,3,0],
    [21,1,78,75,77,5,1],[21,2,80,82,85,3,1],[21,3,77,78,80,4,1],[21,4,79,80,83,2,1],[21,5,75,77,79,3,1],
    [22,1,54,49,55,2,0],[22,2,50,52,50,3,0],[22,3,55,55,58,4,0],[22,4,52,50,52,3,0],[22,5,56,54,57,2,0],
    [23,1,70,72,73,3,1],[23,2,74,75,71,4,1],[23,3,72,78,75,5,1],[23,4,68,70,72,3,1],[23,5,72,74,76,4,1],
    [24,1,48,50,52,2,0],[24,2,52,48,49,3,0],[24,3,50,52,55,4,0],[24,4,49,51,50,2,0],[24,5,53,55,54,3,0],
    [25,1,77,80,82,2,1],[25,2,75,78,80,3,1],[25,3,79,82,85,4,1],[25,4,76,80,81,3,1],[25,5,78,79,83,2,1],
    [26,1,56,54,57,3,0],[26,2,58,57,60,4,0],[26,3,55,55,55,2,0],[26,4,60,58,62,3,0],[26,5,57,56,58,4,0],
    [27,1,72,75,78,4,1],[27,2,74,78,80,3,1],[27,3,70,72,75,5,1],[27,4,73,76,79,2,1],[27,5,75,77,80,3,1],
    [28,1,50,53,55,3,0],[28,2,52,50,52,2,0],[28,3,55,54,57,4,0],[28,4,51,52,54,3,0],[28,5,54,55,58,4,0],
    [29,1,80,82,84,3,1],[29,2,78,80,82,4,1],[29,3,82,84,86,2,1],[29,4,79,82,83,3,1],[29,5,81,83,85,4,1],
    [30,1,46,48,50,2,0],[30,2,50,50,53,3,0],[30,3,48,52,51,4,0],[30,4,51,50,54,2,0],[30,5,52,53,55,3,0],
    [31,1,76,79,80,4,1],[31,2,78,81,83,3,1],[31,3,74,76,78,5,1],[31,4,77,79,81,2,1],[31,5,79,80,82,3,1],
    [32,1,55,57,58,3,0],[32,2,57,58,61,4,0],[32,3,53,55,57,2,0],[32,4,58,60,62,3,0],[32,5,55,57,59,4,0],
    [33,1,73,76,78,3,1],[33,2,75,78,80,4,1],[33,3,71,74,76,5,1],[33,4,74,77,79,2,1],[33,5,76,78,81,3,1],
    [34,1,51,54,56,2,0],[34,2,53,52,54,3,0],[34,3,56,55,58,4,0],[34,4,52,53,55,3,0],[34,5,55,56,59,4,0],
    [35,1,79,81,83,4,1],[35,2,77,80,82,3,1],[35,3,81,83,85,2,1],[35,4,78,81,82,4,1],[35,5,80,82,84,3,1],
    [36,1,47,49,51,2,0],[36,2,51,51,54,3,0],[36,3,49,53,52,4,0],[36,4,52,51,55,2,0],[36,5,53,54,56,3,0],
    [37,1,74,77,79,3,1],[37,2,76,79,81,4,1],[37,3,72,75,77,5,1],[37,4,75,78,80,2,1],[37,5,77,79,82,3,1],
    [38,1,53,56,57,3,0],[38,2,55,57,60,4,0],[38,3,52,54,56,2,0],[38,4,57,59,61,3,0],[38,5,54,56,58,4,0],
    [39,1,71,74,76,4,1],[39,2,73,76,78,3,1],[39,3,69,72,74,5,1],[39,4,72,75,77,2,1],[39,5,74,76,79,3,1],
    [40,1,49,52,54,2,0],[40,2,51,50,52,3,0],[40,3,54,53,56,4,0],[40,4,50,51,53,3,0],[40,5,53,54,57,4,0],
    [41,1,77,80,82,3,1],[41,2,79,82,84,4,1],[41,3,75,77,79,5,1],[41,4,78,81,83,2,1],[41,5,80,82,85,3,1],
    [42,1,54,56,58,3,0],[42,2,56,57,61,4,0],[42,3,53,55,57,2,0],[42,4,58,60,62,3,0],[42,5,55,57,59,4,0],
    [43,1,72,75,77,4,1],[43,2,74,77,79,3,1],[43,3,70,73,75,5,1],[43,4,73,76,78,2,1],[43,5,75,77,80,3,1],
    [44,1,50,53,55,2,0],[44,2,52,51,53,3,0],[44,3,55,54,57,4,0],[44,4,51,52,54,3,0],[44,5,54,55,58,4,0],
    [45,1,78,81,83,3,1],[45,2,76,79,81,4,1],[45,3,80,82,84,2,1],[45,4,77,80,81,4,1],[45,5,79,81,83,3,1],
    [46,1,48,50,52,2,0],[46,2,52,52,55,3,0],[46,3,50,54,53,4,0],[46,4,53,52,56,2,0],[46,5,54,55,57,3,0],
    [47,1,75,78,80,3,1],[47,2,77,80,82,4,1],[47,3,73,76,78,5,1],[47,4,76,79,81,2,1],[47,5,78,80,83,3,1],
    [48,1,55,57,59,3,0],[48,2,57,59,62,4,0],[48,3,54,56,58,2,0],[48,4,59,61,63,3,0],[48,5,56,58,60,4,0],
    [49,1,73,76,78,4,1],[49,2,75,78,80,3,1],[49,3,71,74,76,5,1],[49,4,74,77,79,2,1],[49,5,76,78,81,3,1],
    [50,1,52,54,56,2,0],[50,2,54,53,55,3,0],[50,3,57,56,58,4,0],[50,4,53,54,56,3,0],[50,5,56,57,60,4,0],
]

FEATURES  = ['attendance', 'assignment', 'quiz', 'study_hours']
TIMESTEPS = 5
N_FEAT    = 4

# ══════════════════════════════════════════════════════════════
# AUTO-TRAIN
# ══════════════════════════════════════════════════════════════
def auto_train():
    """Train and save model if files are missing."""
    import numpy as np
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler
    from tensorflow.keras.models   import Sequential
    from tensorflow.keras.layers   import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    df = pd.DataFrame(DATASET, columns=['student_id','week','attendance','assignment','quiz','study_hours','result'])
    seqs, labels = [], []
    for sid, grp in df.groupby('student_id'):
        grp = grp.sort_values('week')
        seqs.append(grp[FEATURES].values)
        labels.append(int(grp['result'].iloc[-1]))

    X = np.array(seqs); y = np.array(labels)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, N_FEAT)).reshape(X_tr.shape)

    mdl = Sequential([
        LSTM(64, input_shape=(TIMESTEPS, N_FEAT), return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1,  activation='sigmoid'),
    ])
    mdl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    mdl.fit(X_tr_s, y_tr, epochs=50, batch_size=8,
            validation_split=0.15, callbacks=[es], verbose=0)

    mdl.save('model.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return mdl, scaler


@st.cache_resource
def load_artifacts():
    if not os.path.exists('model.h5') or not os.path.exists('scaler.pkl'):
        return auto_train()
    from tensorflow.keras.models import load_model
    mdl = load_model('model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return mdl, scaler


# Show spinner only on first run
if not os.path.exists('model.h5') or not os.path.exists('scaler.pkl'):
    with st.spinner("⏳ First launch: Training RNN model... (~30 seconds)"):
        model, scaler = load_artifacts()
else:
    model, scaler = load_artifacts()

# ══════════════════════════════════════════════════════════════
# PREDICT FUNCTION
# ══════════════════════════════════════════════════════════════
def predict_student(weekly_data):
    X = np.array(weekly_data, dtype=float).reshape(1, TIMESTEPS, N_FEAT)
    X_s = scaler.transform(X.reshape(-1, N_FEAT)).reshape(1, TIMESTEPS, N_FEAT)
    pass_prob  = float(model.predict(X_s, verbose=0)[0][0])
    fail_prob  = 1.0 - pass_prob
    prediction = 1 if pass_prob >= 0.5 else 0
    pp = round(pass_prob * 100, 2)
    fp = round(fail_prob * 100, 2)

    if prediction == 1:
        interp = ("🌟 Excellent – very high chance of passing!"  if pp >= 80 else
                  "✅ Good – likely to pass, keep it up!"         if pp >= 65 else
                  "⚠️ Borderline pass – consistent effort needed.")
    else:
        interp = ("❌ High risk of failure – urgent improvement needed." if fp >= 80 else
                  "⚠️ Likely to fail – focus on weak areas now."         if fp >= 65 else
                  "🔶 At risk – small improvements can turn this around.")

    return {"result": prediction, "label": "Pass" if prediction == 1 else "Fail",
            "pass_prob": pp, "fail_prob": fp, "interpretation": interp}

# ══════════════════════════════════════════════════════════════
# UI — HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='text-align:center;color:#5B2C8D;'>🧠 RNN Student Performance Predictor</h1>
<p style='text-align:center;color:gray;'>Powered by LSTM (Long Short-Term Memory) Neural Network</p>
<hr/>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **Recurrent Neural Network (LSTM)**
    that reads 5 weeks of student data as a **sequence**
    and predicts final **Pass / Fail** outcome.

    **Architecture**
    - Input: (5 weeks × 4 features)
    - LSTM Layer 1: 64 units
    - LSTM Layer 2: 32 units
    - Dense Output: Sigmoid
    """)
    st.divider()
    st.subheader("📊 Features per Week")
    st.markdown("""
    - 📅 Attendance %
    - 📝 Assignment marks
    - ❓ Quiz marks
    - ⏰ Study hours
    """)

# ══════════════════════════════════════════════════════════════
# UI — INPUT TABS
# ══════════════════════════════════════════════════════════════
st.subheader("📋 Enter Weekly Student Data (5 Weeks)")
st.info("Fill in the academic data for each week below, then click **Predict**.")

weekly_inputs = []
tabs = st.tabs(["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"])

defaults = {
    "att":  [70, 72, 71, 73, 74],
    "asgn": [75, 76, 78, 77, 80],
    "quiz": [80, 78, 82, 79, 83],
    "sh":   [5,  6,  5,  7,  6 ],
}

for i, tab in enumerate(tabs):
    with tab:
        c1, c2 = st.columns(2)
        with c1:
            att  = st.slider(f"Attendance (%)",    0, 100, defaults["att"][i],  key=f"att_{i}")
            asgn = st.slider(f"Assignment Marks",  0, 100, defaults["asgn"][i], key=f"asgn_{i}")
        with c2:
            quiz = st.slider(f"Quiz Marks",        0, 100, defaults["quiz"][i], key=f"quiz_{i}")
            sh   = st.slider(f"Study Hours/Week",  0,  15, defaults["sh"][i],   key=f"sh_{i}")
        weekly_inputs.append([att, asgn, quiz, sh])

# ── Trend preview ──────────────────────────────────────────────
with st.expander("📈 View Input Trends"):
    chart_df = pd.DataFrame(weekly_inputs,
                            columns=['Attendance','Assignment','Quiz','Study Hours'],
                            index=[f"Wk{i+1}" for i in range(5)])
    st.line_chart(chart_df)

# ══════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════
st.divider()
if st.button("🔮 Predict Performance", type="primary", use_container_width=True):

    with st.spinner("Running LSTM prediction..."):
        r = predict_student(weekly_inputs)

    st.divider()
    st.subheader("📊 Prediction Result")

    if r["result"] == 1:
        st.success(f"## ✅  {r['label']}  —  {r['pass_prob']}% Pass Probability")
    else:
        st.error(f"## ❌  {r['label']}  —  {r['fail_prob']}% Fail Probability")

    st.markdown(f"### {r['interpretation']}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🟢 Pass Probability", f"{r['pass_prob']}%")
        st.progress(r['pass_prob'] / 100)
    with c2:
        st.metric("🔴 Fail Probability", f"{r['fail_prob']}%")
        st.progress(r['fail_prob'] / 100)

    # ── Advice ────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 Personalised Advice")

    avg = lambda col: sum(w[col] for w in weekly_inputs) / 5
    tips = []
    if avg(0) < 70: tips.append("📅 **Low average attendance.** Aim for 75%+ each week.")
    if avg(1) < 60: tips.append("📝 **Assignment marks are weak.** Submit all work on time.")
    if avg(2) < 60: tips.append("❓ **Quiz scores need improvement.** Practice past papers.")
    if avg(3) <  5: tips.append("⏰ **Study hours are low.** Try to study at least 5 hrs/week.")

    # Week-on-week trend warning
    att_trend = weekly_inputs[-1][0] - weekly_inputs[0][0]
    if att_trend < -5:
        tips.append("📉 **Attendance is declining week over week.** Address this urgently.")

    if tips:
        for t in tips: st.warning(t)
    else:
        st.success("🎉 Excellent profile across all weeks! Keep this momentum going.")

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;font-size:12px;'>"
    "RNN Student Predictor · LSTM + Streamlit · Built for Academic AI Projects"
    "</p>", unsafe_allow_html=True
)
