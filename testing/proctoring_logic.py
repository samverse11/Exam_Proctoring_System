import cv2
import dlib
import numpy as np
import face_recognition
from imutils import face_utils
from ultralytics import YOLO

# Load YOLOv8 model for object detection
object_detector = YOLO('yolov8n.pt')

# Load Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load images of known faces (replace with actual candidate images)
image_person1 = face_recognition.load_image_file("person1.jpg")
encoding_person1 = face_recognition.face_encodings(image_person1)[0]

# Store known encodings
known_face_encodings = [encoding_person1]
known_face_names = ["Royan"]

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype="double")

# Add the camera matrix and dist_coeffs here for head pose estimation
focal_length = 1  # Set an appropriate focal length based on your camera
center = (640 / 2, 480 / 2)  # Assuming a 640x480 resolution
camera_matrix = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Logging suspicious activities
def log_suspicious_activity(event, timestamp):
    with open("suspicious_activity_log.txt", "a") as log_file:
        log_file.write(f"{timestamp}: {event}\n")

# Estimate head pose using facial landmarks
def estimate_head_pose(landmarks, frame, camera_matrix, dist_coeffs):
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype="double")

    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Calculate rotation angles (pitch, yaw, roll)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch, yaw, roll = angles
    return pitch, yaw, roll

# Detect gaze direction using eye landmarks
def detect_gaze(landmarks):
    left_eye_pts = landmarks[36:42]
    right_eye_pts = landmarks[42:48]

    left_eye_center = np.mean(left_eye_pts, axis=0).astype("int")
    right_eye_center = np.mean(right_eye_pts, axis=0).astype("int")

    # Horizontal distance between eye centers
    eye_distance = right_eye_center[0] - left_eye_center[0]

    # Gaze direction logic
    if eye_distance < -20:
        return "Looking Right"
    elif eye_distance > 20:
        return "Looking Left"
    else:
        return "Looking Forward"

# Monitor gaze behavior
def monitor_gaze_behavior(gaze_direction, frame_count, frame):
    if gaze_direction != "Looking Forward":
        frame_count["away"] += 1
    else:
        frame_count["away"] = 0

    # If gaze is away for too long, log suspicious activity
    if frame_count["away"] > 30:  # Example: 30 frames (~1 second)
        cv2.putText(frame, "Alert: Suspicious Gaze!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        log_suspicious_activity("Suspicious gaze behavior", timestamp)

# Verify identity using face recognition
def verify_identity(frame, alert_messages):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    current_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, current_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        if name == "Unknown":
            alert_messages.append("Unknown face detected!")  # Append to alerts
            timestamp = cv2.getTickCount() / cv2.getTickFrequency()
            log_suspicious_activity("Unknown face detected", timestamp)

        # Draw face box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# Detect faces, gaze, and head pose estimation
def detect_face_and_behavior(frame, frame_count, camera_matrix, dist_coeffs, alert_messages):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Verify identity using face recognition
    verify_identity(frame, alert_messages)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        gaze_direction = detect_gaze(shape)
        monitor_gaze_behavior(gaze_direction, frame_count, frame)
        cv2.putText(frame, f'Gaze: {gaze_direction}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        pitch, yaw, roll = estimate_head_pose(shape, frame, camera_matrix, dist_coeffs)
        cv2.putText(frame, f'Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}', (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Detect unauthorized objects using YOLOv8
def detect_objects(frame, alert_messages):
    results = object_detector(frame)
    person_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = object_detector.names[class_id]

            if class_name == "person":
                person_count += 1

            if class_name in ["cell phone", "book"]:
                alert_messages.append(f"Unauthorized object detected: {class_name}")  # Append to alerts
                cv2.putText(frame, f"Alert: {class_name} detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if person_count > 1:
        alert_messages.append("Multiple people detected!")  # Append to alerts
        cv2.putText(frame, "Alert: More than one person detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return results[0].plot()  # Annotate frame with detected objects

