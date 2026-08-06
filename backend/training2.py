import os
import cv2
import pickle
import numpy as np
import face_recognition
import dlib

# Base Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODEL_PATH = os.path.join(
    BASE_DIR,
    "shape_predictor_68_face_landmarks.dat"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "encodings.pkl"
)

# Load dlib
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(MODEL_PATH)

# CLAHE
def apply_clahe(image_bgr):

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(2.0, (8, 8))

    l = clahe.apply(l)

    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

# Convert to RGB
def to_rgb(image_bgr):

    return np.ascontiguousarray(
        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        ),
        dtype=np.uint8
    )

# TRAIN FUNCTION
def train_model():

    known_encodings = []

    known_names = []

    print("\nTraining Started...\n")

    # Main Loop
    for person in os.listdir(DATASET_PATH):

        person_path = os.path.join(
            DATASET_PATH,
            person
        )

        if not os.path.isdir(person_path):

            continue

        print(f"\n{person}")

        for img_name in os.listdir(person_path):

            if not img_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                continue

            path = os.path.join(
                person_path,
                img_name
            )

            print(f"{img_name} → ", end="")

            # Load image
            img = cv2.imread(path)

            if img is None:

                print("Image load failed")

                continue

            img = cv2.resize(img, (500, 500))

            # Grayscale
            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            gray = np.array(
                gray,
                dtype=np.uint8
            )

            gray = np.ascontiguousarray(gray)

            try:

                faces = detector(gray, 1)

            except RuntimeError as e:

                print(f"Detector Error: {e}")

                continue

            if len(faces) == 0:

                print("No face detected")

                continue

            try:

                shape = predictor(
                    gray,
                    faces[0]
                )

                landmarks = np.array([
                    [p.x, p.y]
                    for p in shape.parts()
                ])

            except Exception as e:

                print(f"Landmark Error: {e}")

                continue

            # CLAHE
            img = apply_clahe(img)

            # Alignment
            aligned = cv2.resize(img, (224, 224))

            rgb = to_rgb(aligned)

            try:

                encodings = face_recognition.face_encodings(
                    rgb
                )

            except Exception as e:

                print(f"Encoding Error: {e}")

                continue

            if not encodings:

                print("No encoding")

                continue

            known_encodings.append(encodings[0])

            known_names.append(person)

            print("Done")

    print("\n" + "─" * 45)

    # Save Encodings
    if known_encodings:

        with open(OUTPUT_PATH, "wb") as f:

            pickle.dump(
                {
                    "encodings": known_encodings,
                    "names": known_names
                },
                f
            )

        print("Training Complete")

        print(
            f"Saved: {len(known_encodings)} encodings"
        )

    else:

        print("No encodings saved")


# RUN DIRECTLY
if __name__ == "__main__":

    train_model()