import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import time

# Initialize MediaPipe Tasks API
model_path = 'face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

log_file = open("../logs/gaze_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp", "gaze_direction", "face_detected"])
print("Press 'q' to quit.")

# Create the Landmarker and run the loop
with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time()
        
        frame_timestamp_ms = int(timestamp * 1000)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        gaze_direction = "Unknown"
        face_detected = 0

        # API stores landmarks directly as a list of faces in `face_landmarks`
        if results.face_landmarks:
            face_detected = 1
            face_landmarks = results.face_landmarks[0]

            # Left iris and eye landmark indices
            left_iris = face_landmarks[468]
            left_eye_outer = face_landmarks[33]
            left_eye_inner = face_landmarks[133]

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            iris_x = int(left_iris.x * w)
            eye_outer_x = int(left_eye_outer.x * w)
            eye_inner_x = int(left_eye_inner.x * w)

            eye_width = eye_inner_x - eye_outer_x
            
            # Safeguard to prevent division by zero during fast tracking glitches
            if eye_width != 0: 
                iris_position = (iris_x - eye_outer_x) / eye_width

                if iris_position < 0.4:
                    gaze_direction = "Right"
                elif iris_position > 0.6:
                    gaze_direction = "Left"
                else:
                    gaze_direction = "Center"

            # Draw iris point
            cv2.circle(frame, (iris_x, int(left_iris.y * h)), 3, (0, 255, 0), -1)

        csv_writer.writerow([timestamp, gaze_direction, face_detected])

        # Display gaze direction
        cv2.putText(frame, f"Gaze: {gaze_direction}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Gaze Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
log_file.close()
cv2.destroyAllWindows()