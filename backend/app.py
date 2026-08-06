from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import os
import csv
import json
import pickle
import random
import string
import numpy as np
import face_recognition

from datetime import datetime

app = Flask(__name__)

CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")

ENCODING_PATH = os.path.join(BASE_DIR, "encodings.pkl")

STUDENT_FILE = os.path.join(BASE_DIR, "students.json")

SESSION_FILE = os.path.join(BASE_DIR, "session.json")

DOCUMENT_FOLDER = os.path.join(BASE_DIR, "documents")

ATTENDANCE_FOLDER = os.path.join(BASE_DIR, "attendance")

os.makedirs(DOCUMENT_FOLDER, exist_ok=True)

os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)

# LOAD ENCODINGS

with open(ENCODING_PATH, "rb") as f:

    data = pickle.load(f)

known_encodings = data["encodings"]

known_names = data["names"]


# REGISTER API

@app.route("/register", methods=["POST"])
def register():

    name = request.form.get("name")

    reg_no = request.form.get("regNo")

    images = request.files.getlist("images")

    user_folder = os.path.join(DATASET_PATH, name)

    os.makedirs(user_folder, exist_ok=True)

    for idx, image in enumerate(images):

        image.save(
            os.path.join(
                user_folder,
                f"{idx}.jpg"
            )
        )

    students = {}

    if os.path.exists(STUDENT_FILE):

        with open(STUDENT_FILE, "r") as f:

            students = json.load(f)

    students[name] = reg_no

    with open(STUDENT_FILE, "w") as f:

        json.dump(students, f, indent=4)

    return jsonify({
        "message": "Student Registered"
    })


# TRAIN API

@app.route("/train", methods=["POST"])
def train():

    import training2

    training2.train_model()

    global known_encodings
    global known_names

    with open(ENCODING_PATH, "rb") as f:

        data = pickle.load(f)

    known_encodings = data["encodings"]

    known_names = data["names"]

    return jsonify({
        "message": "Training Complete"
    })


# CREATE SESSION API

@app.route("/create-session", methods=["POST"])
def create_session():

    subject = request.form.get("subject")

    teacher = request.form.get("teacher")

    date = request.form.get("date")

    document = request.files["document"]

    access_code = ''.join(
        random.choices(
            string.ascii_uppercase + string.digits,
            k=6
        )
    )

    document.save(
        os.path.join(
            DOCUMENT_FOLDER,
            document.filename
        )
    )

    session_data = {
        "subject": subject,
        "teacher": teacher,
        "date": date,
        "access_code": access_code,
        "document": document.filename
    }

    with open(SESSION_FILE, "w") as f:

        json.dump(session_data, f, indent=4)

    return jsonify({
        "message": "Session Created",
        "access_code": access_code
    })


# SESSION INFO API

@app.route("/session-info", methods=["GET"])
def session_info():

    if os.path.exists(SESSION_FILE):

        with open(SESSION_FILE, "r") as f:

            data = json.load(f)

        return jsonify(data)

    return jsonify({})


# DOWNLOAD DOCUMENT

@app.route("/download-document/<filename>", methods=["GET"])
def download_document(filename):

    return send_file(
        os.path.join(
            DOCUMENT_FOLDER,
            filename
        ),
        as_attachment=True
    )


# MARK ATTENDANCE

def mark_attendance(name):

    with open(SESSION_FILE, "r") as f:

        session = json.load(f)

    subject = session["subject"]

    teacher = session["teacher"]

    date = session["date"]

    csv_name = f"{subject}_{teacher}_{date}.csv"

    csv_path = os.path.join(
        ATTENDANCE_FOLDER,
        csv_name
    )

    students = {}

    if os.path.exists(STUDENT_FILE):

        with open(STUDENT_FILE, "r") as f:

            students = json.load(f)

    reg_no = students.get(name, "Unknown")

    current_time = datetime.now().strftime("%H:%M:%S")

    already_marked = False

    if os.path.exists(csv_path):

        with open(csv_path, "r") as f:

            reader = csv.reader(f)

            next(reader, None)

            for row in reader:

                if row[0] == name:

                    already_marked = True

                    break

    if not already_marked:

        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:

            writer = csv.writer(f)

            if not file_exists:

                writer.writerow([
                    "Name",
                    "RegNo",
                    "Date",
                    "Time"
                ])

            writer.writerow([
                name,
                reg_no,
                date,
                current_time
            ])


# RECOGNIZE API

@app.route("/recognize", methods=["POST"])
def recognize():

    class_code = request.form.get("classCode")

    with open(SESSION_FILE, "r") as f:

        session = json.load(f)

    if class_code != session["access_code"]:

        return jsonify({
            "message": "Invalid Access Code"
        })

    file = request.files["image"]

    image = face_recognition.load_image_file(file)

    face_locations = face_recognition.face_locations(image)

    face_encodings = face_recognition.face_encodings(
        image,
        face_locations
    )

    results = []

    for face_encoding in face_encodings:

        distances = face_recognition.face_distance(
            known_encodings,
            face_encoding
        )

        best_match_index = np.argmin(distances)

        confidence = (
            1 - distances[best_match_index]
        ) * 100

        if confidence > 45:

            name = known_names[best_match_index]

            mark_attendance(name)

        else:

            name = "Unknown"

        results.append({
            "name": name,
            "confidence": round(confidence, 2)
        })

    return jsonify({
        "faces": results
    })


# HISTORY API

@app.route("/attendance-files", methods=["GET"])
def attendance_files():

    files = os.listdir(ATTENDANCE_FOLDER)

    return jsonify(files)

# DOWNLOAD ATTENDANCE FILE

@app.route("/attendance/<filename>", methods=["GET"])
def download_attendance_file(filename):

    return send_file(
        os.path.join(
            ATTENDANCE_FOLDER,
            filename
        ),
        as_attachment=True
    )
if __name__ == "__main__":

    app.run(debug=True)