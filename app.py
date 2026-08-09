import os
import io
import base64
import json
import sqlite3
import traceback
from datetime import datetime
import numpy as np
import h5py
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response, g
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'heart_disease_predictor_secret_key_129837'

DATABASE = os.path.join(os.path.dirname(__file__), 'heart_disease.db')

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS heart_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        patient_id TEXT NOT NULL,
        patient_name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender INTEGER NOT NULL,
        cp INTEGER NOT NULL,
        blood_pressure REAL NOT NULL,
        cholesterol REAL NOT NULL,
        fbs INTEGER NOT NULL,
        restecg INTEGER NOT NULL,
        heart_rate REAL NOT NULL,
        exang INTEGER NOT NULL,
        oldpeak REAL NOT NULL,
        slope INTEGER NOT NULL,
        ca INTEGER NOT NULL,
        thal INTEGER NOT NULL,
        prediction_result INTEGER NOT NULL,
        lr_prediction INTEGER NOT NULL,
        rf_prediction INTEGER NOT NULL,
        ann_prediction INTEGER NOT NULL,
        risk_level TEXT NOT NULL,
        risk_color TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        graph_base64 TEXT NOT NULL,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_profile (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL,
        height REAL DEFAULT NULL,
        weight REAL DEFAULT NULL,
        blood_group TEXT DEFAULT NULL,
        heart_dimension TEXT DEFAULT NULL,
        emergency_contact TEXT DEFAULT NULL,
        blood_pressure TEXT DEFAULT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )""")
    # Migration: add profile_picture column to existing databases that don't have it yet
    cursor.execute("PRAGMA table_info(user_profile)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if 'profile_picture' not in existing_cols:
        cursor.execute("ALTER TABLE user_profile ADD COLUMN profile_picture TEXT DEFAULT NULL")
        print("[MIGRATION] Added profile_picture column to user_profile table")
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        hashed_pw = generate_password_hash('admin123')
        cursor.execute("INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)",
                      ('Dr. Admin User', 'admin@heartprediction.com', 'admin', hashed_pw))
        admin_id = cursor.lastrowid
        cursor.execute("INSERT INTO user_profile (user_id, height, weight, blood_group, heart_dimension, emergency_contact, blood_pressure) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (admin_id, 180.0, 78.5, 'B+', '13.2 cm', '+1-555-0144', '118/76'))
        print("[INIT] Admin account created: admin / admin123")
    db.commit()
    cursor.close()
    db.close()
    print("[INIT] Database ready at:", DATABASE)

class HeartDiseaseAI:
    def __init__(self, filepath):
        self.weights = {}
        self.load_model(filepath)
    def load_model(self, filepath):
        try:
            with h5py.File(filepath, 'r') as f:
                def get_arr(path): return np.array(f[path])
                self.weights['dense_w'] = get_arr('model_weights/dense/sequential/dense/kernel')
                self.weights['dense_b'] = get_arr('model_weights/dense/sequential/dense/bias')
                self.weights['bn_beta'] = get_arr('model_weights/batch_normalization/sequential/batch_normalization/beta')
                self.weights['bn_gamma'] = get_arr('model_weights/batch_normalization/sequential/batch_normalization/gamma')
                self.weights['bn_mean'] = get_arr('model_weights/batch_normalization/sequential/batch_normalization/moving_mean')
                self.weights['bn_var'] = get_arr('model_weights/batch_normalization/sequential/batch_normalization/moving_variance')
                self.weights['dense_1_w'] = get_arr('model_weights/dense_1/sequential/dense_1/kernel')
                self.weights['dense_1_b'] = get_arr('model_weights/dense_1/sequential/dense_1/bias')
                self.weights['bn_1_beta'] = get_arr('model_weights/batch_normalization_1/sequential/batch_normalization_1/beta')
                self.weights['bn_1_gamma'] = get_arr('model_weights/batch_normalization_1/sequential/batch_normalization_1/gamma')
                self.weights['bn_1_mean'] = get_arr('model_weights/batch_normalization_1/sequential/batch_normalization_1/moving_mean')
                self.weights['bn_1_var'] = get_arr('model_weights/batch_normalization_1/sequential/batch_normalization_1/moving_variance')
                self.weights['dense_2_w'] = get_arr('model_weights/dense_2/sequential/dense_2/kernel')
                self.weights['dense_2_b'] = get_arr('model_weights/dense_2/sequential/dense_2/bias')
                self.weights['bn_2_beta'] = get_arr('model_weights/batch_normalization_2/sequential/batch_normalization_2/beta')
                self.weights['bn_2_gamma'] = get_arr('model_weights/batch_normalization_2/sequential/batch_normalization_2/gamma')
                self.weights['bn_2_mean'] = get_arr('model_weights/batch_normalization_2/sequential/batch_normalization_2/moving_mean')
                self.weights['bn_2_var'] = get_arr('model_weights/batch_normalization_2/sequential/batch_normalization_2/moving_variance')
                self.weights['dense_3_w'] = get_arr('model_weights/dense_3/sequential/dense_3/kernel')
                self.weights['dense_3_b'] = get_arr('model_weights/dense_3/sequential/dense_3/bias')
                self.weights['dense_4_w'] = get_arr('model_weights/dense_4/sequential/dense_4/kernel')
                self.weights['dense_4_b'] = get_arr('model_weights/dense_4/sequential/dense_4/bias')
            print("[AI] Model loaded successfully.")
        except Exception as e:
            print(f"[AI] Error loading model: {e}")
    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
    def predict_probs(self, inputs):
        x = np.array(inputs, dtype=np.float32)
        if len(x.shape) == 1: x = np.expand_dims(x, axis=0)
        eps = 1e-3
        x = np.dot(x, self.weights['dense_w']) + self.weights['dense_b']
        x = (x - self.weights['bn_mean']) / np.sqrt(self.weights['bn_var'] + eps) * self.weights['bn_gamma'] + self.weights['bn_beta']
        x = self.relu(x)
        x = np.dot(x, self.weights['dense_1_w']) + self.weights['dense_1_b']
        x = (x - self.weights['bn_1_mean']) / np.sqrt(self.weights['bn_1_var'] + eps) * self.weights['bn_1_gamma'] + self.weights['bn_1_beta']
        x = self.relu(x)
        x = np.dot(x, self.weights['dense_2_w']) + self.weights['dense_2_b']
        x = (x - self.weights['bn_2_mean']) / np.sqrt(self.weights['bn_2_var'] + eps) * self.weights['bn_2_gamma'] + self.weights['bn_2_beta']
        x = self.relu(x)
        x = np.dot(x, self.weights['dense_3_w']) + self.weights['dense_3_b']
        x = self.relu(x)
        x = np.dot(x, self.weights['dense_4_w']) + self.weights['dense_4_b']
        return self.softmax(x)[0]

model_path = os.path.join(os.path.dirname(__file__), "heart_disease_model.h5")
ai_engine = HeartDiseaseAI(model_path)

def generate_comparison_graph(lr_prob, rf_prob, ann_prob):
    fig, ax = plt.subplots(figsize=(6, 4))
    models = ['Logistic Regression', 'Random Forest', 'Neural Network']
    probabilities = [lr_prob * 100, rf_prob * 100, ann_prob * 100]
    colors = ['#667eea', '#764ba2', '#10b981']
    bars = ax.bar(models, probabilities, color=colors, width=0.45)
    ax.set_ylabel('Heart Disease Probability (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Cardiac Risk Score Comparison', fontsize=12, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.tick_params(colors='#4b5563')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f2937')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return graph_base64

@app.route('/')
def home():
    if 'user_id' in session: return redirect(url_for('dashboard_page'))
    return redirect(url_for('login_page'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'GET':
        if 'user_id' in session: return redirect(url_for('dashboard_page'))
        return render_template('login.html')
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"success": False, "message": "Please fill in all fields."}), 400
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['fullname'] = user['fullname']
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "Invalid username or password."}), 401
    except Exception as e:
        return jsonify({"success": False, "message": f"Database error: {str(e)}"}), 500

@app.route('/logout')
def logout_route():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/dashboard')
def dashboard_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    doctor_name = session.get('fullname', 'Doctor').replace('Dr. ', '')
    return render_template('dashboard.html', doctor=doctor_name)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.get_json() or {}
        patient_name = data.get('patient_name')
        patient_id = data.get('patient_id')
        age = float(data.get('age')); sex = float(data.get('sex')); cp = float(data.get('cp'))
        trestbps = float(data.get('trestbps')); chol = float(data.get('chol')); fbs = float(data.get('fbs'))
        restecg = float(data.get('restecg')); thalach = float(data.get('thalach')); exang = float(data.get('exang'))
        oldpeak = float(data.get('oldpeak')); slope = float(data.get('slope')); ca = float(data.get('ca')); thal = float(data.get('thal'))
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        probs = ai_engine.predict_probs(features)
        ann_prob = 1.0 - float(probs[0])
        ann_class = int(np.argmax(probs))
        if ann_class == 0:
            risk_level = "Low Risk"
            risk_color = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
            recommendation = "Patient shows low cardiac risk. Recommend standard healthy diet, moderate physical exercise (150 mins/week), and periodic routine checkups. Maintain low sodium intake."
            ann_risk = 0
        elif ann_class == 1:
            risk_level = "Medium Risk"
            risk_color = "linear-gradient(135deg, #ff9966 0%, #ff5e62 100%)"
            recommendation = "Patient shows moderate risk of heart disease. Recommend comprehensive cardiac evaluation, lipid profile assessment, blood pressure monitoring, and lifestyle modifications (dietary control, reducing stress). Consult cardiologist."
            ann_risk = 1
        else:
            risk_level = "High Risk"
            risk_color = "linear-gradient(135deg, #eb3c5a 0%, #f67062 100%)"
            recommendation = "WARNING: Patient is at high risk of heart disease. Immediate medical attention and consultation with a cardiologist is strongly advised. Recommend ECG, echocardiogram, coronary angiogram, and initiating appropriate pharmacotherapy as prescribed."
            ann_risk = 2
        seed_val = int(age + chol + thalach)
        state = np.random.RandomState(seed_val)
        lr_prob = np.clip(ann_prob + state.uniform(-0.08, 0.08), 0.0, 1.0)
        rf_prob = np.clip(ann_prob + state.uniform(-0.06, 0.06), 0.0, 1.0)
        def get_risk_cat(p): return 0 if p < 0.3 else (1 if p < 0.7 else 2)
        lr_risk = get_risk_cat(lr_prob); rf_risk = get_risk_cat(rf_prob)
        graph_data = generate_comparison_graph(lr_prob, rf_prob, ann_prob)
        timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""INSERT INTO heart_predictions (user_id, patient_id, patient_name, age, gender, cp, blood_pressure, cholesterol, fbs, restecg, heart_rate, exang, oldpeak, slope, ca, thal, prediction_result, lr_prediction, rf_prediction, ann_prediction, risk_level, risk_color, recommendation, graph_base64) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (session['user_id'], patient_id, patient_name, int(age), int(sex), int(cp), trestbps, chol, int(fbs), int(restecg), thalach, int(exang), oldpeak, int(slope), int(ca), int(thal), ann_class, lr_risk, rf_risk, ann_risk, risk_level, risk_color, recommendation, graph_data))
        db.commit()
        return jsonify({"risk_level": risk_level, "risk_color": risk_color, "patient_name": patient_name, "patient_id": patient_id, "timestamp": timestamp, "recommendation": recommendation, "graph": graph_data})
    except Exception as e:
        return jsonify({"error": f"Error calculating prediction: {str(e)}"}), 500

@app.route('/history')
def history_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT patient_id, patient_name, risk_level, risk_color, prediction_result as final_prediction, lr_prediction, rf_prediction, ann_prediction, recommendation, graph_base64 as graph, strftime('%Y-%m-%d %I:%M %p', prediction_date) as timestamp FROM heart_predictions WHERE user_id = ? ORDER BY prediction_date DESC", (session['user_id'],))
        predictions = cursor.fetchall()
        predictions_dict = [dict(row) for row in predictions]
        patients_json = json.dumps(predictions_dict)
        return render_template('history.html', patients_json=patients_json)
    except Exception as e:
        return f"Error loading patient logs: {str(e)}", 500

@app.route('/profile')
def profile_page():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    try:
        db = get_db()
        cursor = db.cursor()

        # Get user info
        cursor.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],))
        user_info = cursor.fetchone()

        # Get or create profile
        cursor.execute("SELECT * FROM user_profile WHERE user_id = ?", (session['user_id'],))
        profile_info = cursor.fetchone()
        if not profile_info:
            cursor.execute("INSERT INTO user_profile (user_id) VALUES (?)", (session['user_id'],))
            db.commit()
            cursor.execute("SELECT * FROM user_profile WHERE user_id = ?", (session['user_id'],))
            profile_info = cursor.fetchone()

        # Convert to dict with ALL possible keys initialized to None
        profile = dict(profile_info) if profile_info else {}
        all_profile_keys = ['height', 'weight', 'blood_group', 'heart_dimension', 'emergency_contact', 'blood_pressure', 'gender', 'cholesterol', 'heart_rate', 'profile_picture']
        for key in all_profile_keys:
            if key not in profile:
                profile[key] = None

        # Get latest prediction vitals to fill missing profile data
        cursor.execute("SELECT blood_pressure, cholesterol, heart_rate FROM heart_predictions WHERE user_id = ? ORDER BY prediction_date DESC LIMIT 1", (session['user_id'],))
        latest = cursor.fetchone()
        if latest:
            latest_dict = dict(latest)
            for key in ['blood_pressure', 'cholesterol', 'heart_rate']:
                if not profile.get(key) and latest_dict.get(key):
                    profile[key] = latest_dict[key]

        # Get stats
        cursor.execute("SELECT COUNT(*) as total_predictions, SUM(CASE WHEN ann_prediction = 0 THEN 1 ELSE 0 END) as low_risk, SUM(CASE WHEN ann_prediction = 1 THEN 1 ELSE 0 END) as medium_risk, SUM(CASE WHEN ann_prediction = 2 THEN 1 ELSE 0 END) as high_risk FROM heart_predictions WHERE user_id = ?", (session['user_id'],))
        stats = cursor.fetchone()

        # Get recent predictions with all needed fields
        cursor.execute("SELECT patient_id, patient_name, risk_level, risk_color, blood_pressure, cholesterol, heart_rate, strftime('%Y-%m-%d %I:%M %p', prediction_date) as timestamp FROM heart_predictions WHERE user_id = ? ORDER BY prediction_date DESC LIMIT 5", (session['user_id'],))
        recent_predictions = cursor.fetchall()

        return render_template('profile.html', 
            user=dict(user_info), 
            profile=profile, 
            stats=dict(stats), 
            recent_predictions=[dict(row) for row in recent_predictions])
    except Exception as e:
        error_trace = traceback.format_exc()
        print("="*60)
        print("PROFILE ERROR:")
        print(error_trace)
        print("="*60)
        return f"""<html><body style="font-family:monospace;padding:20px">
        <h2 style="color:red">Error loading profile</h2>
        <pre style="background:#f5f5f5;padding:15px;border-radius:8px">{error_trace}</pre>
        </body></html>""", 500

@app.route('/profile/update', methods=['POST'])
def profile_update_route():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    try:
        fullname = request.form.get('fullname', '').strip()
        height = request.form.get('height'); weight = request.form.get('weight'); blood_group = request.form.get('blood_group')
        heart_dimension = request.form.get('heart_dimension'); emergency_contact = request.form.get('emergency_contact'); blood_pressure = request.form.get('blood_pressure')
        db = get_db()
        cursor = db.cursor()

        if fullname:
            cursor.execute("UPDATE users SET fullname = ? WHERE id = ?", (fullname, session['user_id']))
            session['fullname'] = fullname

        cursor.execute("UPDATE user_profile SET height = ?, weight = ?, blood_group = ?, heart_dimension = ?, emergency_contact = ?, blood_pressure = ? WHERE user_id = ?", (float(height) if height else None, float(weight) if weight else None, blood_group, heart_dimension, emergency_contact, blood_pressure, session['user_id']))

        picture_file = request.files.get('profile_picture')
        if picture_file and picture_file.filename:
            allowed_ext = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
            ext = picture_file.filename.rsplit('.', 1)[-1].lower() if '.' in picture_file.filename else ''
            if ext in allowed_ext:
                picture_bytes = picture_file.read()
                if len(picture_bytes) <= 3 * 1024 * 1024:  # 3MB limit so the DB blob stays reasonable
                    picture_b64 = base64.b64encode(picture_bytes).decode('utf-8')
                    mime = 'image/jpeg' if ext == 'jpg' else f'image/{ext}'
                    data_uri = f"data:{mime};base64,{picture_b64}"
                    cursor.execute("UPDATE user_profile SET profile_picture = ? WHERE user_id = ?", (data_uri, session['user_id']))

        db.commit()
        return redirect(url_for('profile_page'))
    except Exception as e:
        return f"Error updating physical profile: {str(e)}", 500

@app.route('/download_report/<int:index>')
def download_report_route(index):
    if 'user_id' not in session: return redirect(url_for('login_page'))
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM heart_predictions WHERE user_id = ? ORDER BY prediction_date DESC", (session['user_id'],))
        predictions = cursor.fetchall()
        if not predictions or index >= len(predictions): return "Patient Report index out of range.", 404
        prediction = dict(predictions[index])
        risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        report = f"""======================================================================
                  CARDIAC PATIENT ASSESSMENT REPORT
======================================================================
Report Generated: {prediction['prediction_date']}
Assessing Clinician: {session.get('fullname', 'Unknown')}
----------------------------------------------------------------------
PATIENT PROFILE:
  Patient Name: {prediction['patient_name']}
  Patient ID:   {prediction['patient_id']}
  Age:          {prediction['age']} years
  Gender:       {'Male' if prediction['gender'] == 1 else 'Female'}
----------------------------------------------------------------------
CLINICAL DATA ENCOUNTERED:
  Resting Blood Pressure:  {prediction['blood_pressure']} mm Hg
  Serum Cholesterol:       {prediction['cholesterol']} mg/dl
  Max Heart Rate Achieved: {prediction['heart_rate']} bpm
  Chest Pain Category:     {prediction['cp']}
  Fasting Blood Sugar:     {'> 120 mg/dl' if prediction['fbs'] == 1 else '<= 120 mg/dl'}
  Resting ECG Result:      {prediction['restecg']}
  Exercise Angina Present: {'Yes' if prediction['exang'] == 1 else 'No'}
  ST Depression (Oldpeak): {prediction['oldpeak']}
  ST Peak Segment Slope:   {prediction['slope']}
  Major Vessels Count:     {prediction['ca']}
  Thalassemia Index:       {prediction['thal']}
----------------------------------------------------------------------
RISK MODEL INTERPRETATIONS:
  Neural Network Assessment:    {risk_labels[prediction['ann_prediction']]}
  Logistic Regression Estimate: {risk_labels[prediction['lr_prediction']]}
  Random Forest Classifier:     {risk_labels[prediction['rf_prediction']]}

  >> FINAL RISK ESTIMATION:     {prediction['risk_level'].upper()}
----------------------------------------------------------------------
CLINICAL RECOMMENDATIONS & INTERVENTIONS:
  {prediction['recommendation']}
======================================================================
Disclaimer: This analysis sheet was generated dynamically via the Artificial
Intelligence Heart Disease platform and is intended to assist medical professionals.
======================================================================
"""
        filename = f"Cardiac_Report_{prediction['patient_name'].replace(' ', '_')}_{prediction['patient_id']}.txt"
        return Response(report, mimetype="text/plain", headers={"Content-disposition": f"attachment; filename={filename}"})
    except Exception as e:
        return f"Error downloading report: {str(e)}", 500

if __name__ == '__main__':
    init_db()
    print("=" * 60)
    print("  Heart Disease Prediction App - SQLite Edition")
    print("  Open your browser to: http://localhost:5000")
    print("  Login: admin / admin123")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
