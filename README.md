# AI-Powered Online Exam Proctoring Platform

A secure, intelligent online exam system combining live proctoring with behavior monitoring to ensure academic integrity. This platform provides a smooth and scalable way to conduct online tests while deterring cheating using real-time face and object detection.

## ✅ Features at a Glance

-🎥 Real-time webcam stream monitoring

-🧠 Face recognition using face_recognition and dlib

-👥 Alert if multiple faces are detected

-📵 Phone detection using YOLOv8 object detection

-⛔ Exam termination upon suspicious activity

-📝 Activity logs saved to suspicious_activity_log.txt

-🔐 Role-based login (Student, Admin/Proctor)

-⚙️ Django-admin interface for exam setup and user management

## 💡 Motivation

With the shift to remote education, ensuring exam honesty has become a growing challenge. Manual proctoring is tedious and limited. This platform leverages AI to offer:

-🚫 Automated violation detection

-🔍 Real-time monitoring without human invigilators

-🔄 Smooth integration with frontend/backend

## 🛠 Tech Stack


| 🧩 Component        | ⚙️ Tech Used                                      |
|---------------------|--------------------------------------------------|
| 🎨 Frontend         | HTML, CSS                                        |
| 🐍 Backend          | Django (Python)                                  |
| 🤖 AI Proctoring    | OpenCV, dlib, face_recognition, YOLOv8           |
| 🗃️ Database         | SQLite (default Django DB)                       |
| 🚀 Deployment       | Localhost                                        |


## ⚙️ Installation Guide (Tested on Python 3.10.8)

⚠️ Note: dlib requires Python 3.10.8 and CMake ≥ 3.22. Make sure these are properly installed.

1️⃣ Clone the Repository

git clone https://github.com/samverse11/Exam_Proctoring_System.git
cd YOUR_REPO_NAME

2️⃣ Set Up Virtual Environment

python -m venv venv
./venv/Scripts/activate    # On Windows
### OR
source venv/bin/activate  # On Unix/macOS

3️⃣ Install Requirements

pip install -r requirements.txt

💡 If you face issues with dlib, manually build it:

cd testing/dlib
python setup.py install

## 🧪 Django Setup

Run the Following:

python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser

## 🚀 Running the System

### 🖥 Terminal 1: Start Django Server

python manage.py runserver

### 📷 Terminal 2: Launch Proctoring Module

cd testing
python app.py

🎯 When a student begins the exam, the proctoring window opens and starts real-time monitoring using webcam feed.

## 📂 Important Notes on Files

Ignored by Git (.gitignore):

venv/ – your virtual environment

__pycache__/ – Python cache files

db.sqlite3 – database file (auto-generated after migrations)

media/ – stores profile pictures and uploads


## 📝 Logs & Reports

All suspicious activity is recorded in:

testing/suspicious_activity_log.txt

You can configure thresholds or alert triggers via app.py

