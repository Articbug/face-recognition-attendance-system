import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime

# ==============================
# LOAD TRAINED ENCODINGS
# ==============================

ENCODING_PATH = r"C:\Users\chand\Documents\Projects\Face-Recognize-Model\3. Models\backend\encodings.pkl"

with open(ENCODING_PATH, "rb") as f:
    data = pickle.load(f)

# ==============================
# SETTINGS
# ==============================

THRESHOLD = 0.48

attendance = set()

# ==============================
# CLAHE FUNCTION
# ==============================

def apply_clahe(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

# ==============================
# ATTENDANCE FUNCTION
# ==============================

def mark_attendance(name):

    if name not in attendance:

        attendance.add(name)

        now = datetime.now()

        date = now.strftime("%d-%m-%Y")
        time = now.strftime("%H:%M:%S")

        CSV_PATH = r"C:\Users\chand\Documents\Projects\Face-Recognize-Model\3. Models\backend\attendance.csv"

        with open(CSV_PATH, "a") as f:
            f.write(f"{name},{date},{time}\n")

        print("✅ Attendance Saved:", name)

# ==============================
# START CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

print("✅ Face Recognition Started")
print("Press 'q' to Quit")

frame_count = 0

# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    frame_count += 1

    # Process every 2nd frame
    if frame_count % 2 != 0:
        continue

    # Resize frame for speed
    small = cv2.resize(
        frame,
        (0, 0),
        fx=0.5,
        fy=0.5
    )

    # Apply CLAHE
    small = apply_clahe(small)

    # Convert to RGB
    rgb = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2RGB
    )

    rgb = np.ascontiguousarray(
        rgb,
        dtype=np.uint8
    )

    # Detect faces
    faces = face_recognition.face_locations(rgb)

    # Generate encodings
    encodings = face_recognition.face_encodings(
        rgb,
        faces
    )

    # ==============================
    # FACE MATCHING
    # ==============================

    for encoding, (top, right, bottom, left) in zip(encodings, faces):

        name = "Unknown"

        distances = face_recognition.face_distance(
            data["encodings"],
            encoding
        )

        if len(distances) > 0:

            idx = np.argmin(distances)

            min_distance = distances[idx]

            print("Distance:", min_distance)

            if min_distance < THRESHOLD:

                detected_name = data["names"][idx]

                name = f"{detected_name} ({min_distance:.2f})"

                mark_attendance(detected_name)

            else:
                name = "Unknown"

        # Restore original frame size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw rectangle
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Show name
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show webcam
    cv2.imshow(
        "Face Recognition Attendance System",
        frame
    )

    # Quit button
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()