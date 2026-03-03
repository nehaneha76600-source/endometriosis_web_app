from flask import Flask, render_template, request, redirect, session, url_for
import sqlite3
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
import gdown

MODEL_PATH = "models/cnn_model.keras"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    url = "https://drive.google.com/uc?id=11-xIcteY6qN00AqXPlAh9biskZBqAfS0"
    gdown.download(url, MODEL_PATH, quiet=False)

DOCTOR_SECRET_CODE = "ENDO2026"

app = Flask(__name__)
app.secret_key = "secure_medical_key"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Models
ml_model = joblib.load("models/ml_model.pkl")
scaler = joblib.load("models/scaler.pkl")
cnn_model = load_model("models/cnn_model.keras")
# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    # Create users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT,
        status TEXT DEFAULT 'approved',
        email TEXT,
        verified INTEGER DEFAULT 0
    )
    """)

    # Create records table
    c.execute("""
    CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        age REAL,
        height REAL,
        weight REAL,
        bmi REAL,
        heavy_bleeding INTEGER,
        irregular_periods INTEGER,
        pelvic_pain INTEGER,
        hormonal_symptoms INTEGER,
        ml_score REAL,
        cnn_score REAL,
        final_score REAL,
        risk_category TEXT,
        image_path TEXT
    )
    """)

    # Create default admin if not exists
    c.execute("SELECT * FROM users WHERE role='admin'")
    admin = c.fetchone()

    if not admin:
        hashed = generate_password_hash("admin123")
        c.execute("""
        INSERT INTO users(username,password,role,status,email,verified)
        VALUES(?,?,?,?,?,?)
        """, ("admin", hashed, "admin", "approved", "admin@gmail.com", 1))

        print("Admin created: admin / admin123")

    conn.commit()
    conn.close()

# ---------------- RISK LOGIC ----------------
def categorize_risk(prob):
    percent = prob * 100
    if percent >= 70:
        return "High Risk"
    elif percent >= 40:
        return "Moderate Risk"
    else:
        return "Low Risk"

# ---------------- LANDING PAGE ----------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        email = request.form['email']
        secret_code = request.form.get('secret_code')

        hashed_password = generate_password_hash(password)

        if role == "doctor":
            if secret_code != DOCTOR_SECRET_CODE:
                return "Invalid Doctor Secret Code"
            status = "pending"
            verified = 0
        else:
            status = "approved"
            verified = 1

        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("INSERT INTO users(username,password,role,status,email,verified) VALUES(?,?,?,?,?,?)",
                  (username, hashed_password, role, status, email, verified))

        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")
# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        # Now selecting all required fields
        c.execute("SELECT id, password, role, status, verified FROM users WHERE username=?",
                  (username,))
        user = c.fetchone()

        conn.close()

        if user and check_password_hash(user[1], password):

            # Check account approval
            if user[3] != "approved":
                return "Your account is not approved by admin yet."

            # Check email verification
            if user[4] == 0:
                return "Please verify your email first."

            # Store session
            session["user_id"] = user[0]
            session["username"] = username
            session["role"] = user[2]

            # Redirect based on role
            if user[2] == "patient":
                return redirect("/patient")

            elif user[2] == "doctor":
                return redirect("/doctor")

            elif user[2] == "admin":
                return redirect("/admin")

        else:
            return "Invalid Username or Password"

    return render_template("login.html")

# ---------------- PATIENT DASHBOARD ----------------
@app.route("/patient")
def patient():
    if session.get("role") not in ["patient", "admin"]:
        return redirect("/login")
    return render_template("patient_dashboard.html")

@app.route("/patient_predict", methods=["POST"])
def patient_predict():

    age = float(request.form["age"])
    height = float(request.form["height"])
    weight = float(request.form["weight"])
    bmi = weight / ((height/100)**2)

    heavy = int(request.form["heavy_bleeding"])
    irregular = int(request.form["irregular_periods"])
    pelvic = int(request.form["pelvic_pain"])
    hormonal = int(request.form["hormonal_symptoms"])
    pain_intercourse = int(request.form["pain_during_intercourse"])
    family_history = int(request.form["family_history"])

    features = np.array([[age,height,weight,bmi,
                      heavy,irregular,pelvic,hormonal,
                      pain_intercourse,family_history]]) 


    scaled = scaler.transform(features)
    ml_prob = ml_model.predict_proba(scaled)[0][1]

    risk = categorize_risk(ml_prob)

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""INSERT INTO records(username,age,height,weight,bmi,
                heavy_bleeding,irregular_periods,pelvic_pain,hormonal_symptoms,
                ml_score,risk_category)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (session["username"],age,height,weight,bmi,
                 heavy,irregular,pelvic,hormonal,
                 ml_prob,risk))
    conn.commit()
    conn.close()

    return render_template("result.html",
                           ml=round(ml_prob*100,2),
                           risk=risk)

# ---------------- PATIENT HISTORY ----------------
@app.route("/history")
def history():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM records WHERE username=?",
              (session["username"],))
    data = c.fetchall()
    conn.close()
    return render_template("patient_history.html", data=data)

# ---------------- DOCTOR DASHBOARD ----------------
@app.route("/doctor")
def doctor():
    if session.get("role") not in ["doctor", "admin"]:
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM records")
    data = c.fetchall()
    conn.close()

    return render_template("doctor_dashboard.html", data=data)

@app.route("/doctor_view/<int:id>")
def doctor_view(id):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM records WHERE id=?",(id,))
    record = c.fetchone()
    conn.close()
    return render_template("doctor_patient_view.html", record=record)

@app.route("/doctor_upload/<int:id>", methods=["POST"])
def doctor_upload(id):

    # 1️⃣ Get uploaded file
    file = request.files['image']

    if file.filename == "":
        return "No file selected"

    # 2️⃣ Save file
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    path = os.path.join(upload_folder, file.filename)
    file.save(path)

    # 3️⃣ Get model input size automatically
    input_shape = cnn_model.input_shape
    target_height = input_shape[1]
    target_width = input_shape[2]

    # 4️⃣ Load and resize image
    img = image.load_img(path, target_size=(target_height, target_width))

    # 5️⃣ Convert to array
    img_array = image.img_to_array(img)

    # 6️⃣ Normalize
    img_array = img_array / 255.0

    # 7️⃣ Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # 8️⃣ Predict
    cnn_prob = float(cnn_model.predict(img_array)[0][0])

    # 9️⃣ Get ML score from database
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT ml_score FROM records WHERE id=?", (id,))
    ml_prob = c.fetchone()[0]

    # 🔟 Combine ML + CNN
    final = (ml_prob + cnn_prob) / 2
    risk = categorize_risk(final)

    # 1️⃣1️⃣ Update record
    c.execute("""
        UPDATE records
        SET cnn_score=?, final_score=?, risk_category=?, image_path=?
        WHERE id=?
    """, (cnn_prob, final, risk, path, id))

    conn.commit()
    conn.close()

    return redirect("/doctor")

@app.route("/admin")
def admin_panel():
    if session.get("role") != "admin":
        return "Unauthorized"

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE role='doctor' AND status='pending'")
    doctors = c.fetchall()
    conn.close()

    return render_template("admin.html", doctors=doctors)


@app.route("/approve_doctor/<int:id>")
def approve_doctor(id):
    if session.get("role") != "admin":
        return "Unauthorized"

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("UPDATE users SET status='approved', verified=1 WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect("/admin")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)