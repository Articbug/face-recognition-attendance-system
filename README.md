# 🎓 KYC-Integrated Deep Learning Based Face Recognition System for Secure Class-Wise Attendance Management

## 📖 Overview

This project is a web-based smart attendance management system that uses deep learning-based face recognition to automate classroom attendance. The system combines facial authentication with session-based access control to prevent proxy attendance and improve classroom security.

Students register by capturing multiple facial images, which are processed to generate facial embeddings using the **dlib ResNet** model. During attendance, a teacher-generated access code must be entered before facial verification is performed. Upon successful recognition, attendance is automatically recorded with the student's name, registration number, subject, date, and timestamp.

---

## ✨ Features

* 🔐 KYC-based student registration
* 😊 Real-time face recognition using webcam
* 🎯 Deep learning facial embeddings (dlib ResNet)
* 🔑 Random classroom access code generation
* 👨‍🏫 Admin panel for session management
* 📄 Classroom document/notes sharing
* 📊 Attendance history with CSV export
* 🌙 CLAHE image enhancement for improved low-light recognition
* ⚡ Average recognition time below 2 seconds

---

## 🛠 Technologies Used

### Frontend

* React.js
* Tailwind CSS
* Axios
* React Router DOM
* Lucide React

### Backend

* Python 3.10
* Flask
* Flask-CORS

### Computer Vision & AI

* OpenCV
* dlib
* face_recognition
* NumPy

### Storage

* CSV Files
* JSON

### Development Tools

* Visual Studio Code
* Git & GitHub

---

## 🏗 System Architecture

```text
Student Registration
        │
        ▼
 Image Capture
        │
        ▼
 CLAHE Preprocessing
        │
        ▼
 Face Embedding Generation
        │
        ▼
 Store Facial Encodings
        │
        ▼
────────────────────────────────────

Teacher Creates Session
        │
        ▼
Generate Random Access Code
        │
        ▼
Student Enters Access Code
        │
        ▼
Live Face Recognition
        │
        ▼
Attendance Verification
        │
        ▼
Attendance Saved in CSV
```

---

## 📂 Project Structure

```text
face-recognition-attendance-system
│
├── frontend
│   ├── src
│   ├── public
│   └── package.json
│
├── backend
│   ├── app.py
│   ├── training2.py
│   ├── recognize2.py
│   ├── attendance
│   ├── dataset
│   └── models
│
├── screenshots
│
├── README.md
└── .gitignore
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/face-recognition-attendance-system.git
```

### Backend

```bash
cd backend

pip install flask
pip install flask-cors
pip install opencv-python
pip install face_recognition
pip install dlib
pip install numpy
pip install pandas

python app.py
```

### Frontend

```bash
cd frontend

npm install

npm run dev
```

---

## 📷 Screenshots

### Registration Page

*(Add Screenshot Here)*

### Attendance Page

*(Add Screenshot Here)*

### Admin Panel

*(Add Screenshot Here)*

### Attendance History

*(Add Screenshot Here)*

---

## 📈 Performance

| Metric                | Result      |
| --------------------- | ----------- |
| Recognition Accuracy  | 96%         |
| Tested Subjects       | 50          |
| Correct Recognitions  | 48          |
| Average Response Time | < 2 Seconds |
| Preprocessing         | CLAHE       |
| Recognition Engine    | dlib ResNet |

---

## 🔮 Future Enhancements

* Cloud Deployment
* Database Integration
* Mobile Application
* Liveness Detection
* Multi-Camera Support
* AI-Based Classroom Analytics

---

## 👨‍💻 Author

**Chandan Sahoo**

Bachelor of Technology (Computer Science & Engineering)

Institute of Technical Education and Research (ITER)

Siksha 'O' Anusandhan (Deemed to be University)

---

## 📜 License

This project was developed as part of the Final Year Research Project (FYRP 2026) for academic purposes.
