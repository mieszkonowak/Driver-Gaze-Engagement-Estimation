import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import time
import tkinter as tk
import os
from PIL import Image, ImageTk 

# Initialize MediaPipe Tasks API
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'face_landmarker.task')

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

# Global variables
cap = None
landmarker = None
log_file = None
csv_writer = None
is_tracking = False
yawn_start_time = None
yawn_triggered = False
perclos_buffer = []
tracker_start_time = None

def start_tracker():
    global cap, landmarker, log_file, csv_writer, is_tracking, yawn_start_time, yawn_triggered, perclos_buffer, tracker_start_time

    yawn_start_time = None
    yawn_triggered = False
    perclos_buffer = [] 
    tracker_start_time = time.time() 
    
    if is_tracking: 
        return 

    # Setup files and camera
    log_path = os.path.join(script_dir, "..", "logs", "gaze_log.csv")
    log_file = open(log_path, mode="w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["timestamp", "gaze_direction", "yawn_status", "blink_status", "face_detected", "ear_value", "mar_value", "perclos_score"])
    
    landmarker = FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    is_tracking = True

    process_frame()

def process_frame():
    global cap, landmarker, log_file, csv_writer, is_tracking, yawn_start_time, yawn_triggered, perclos_buffer, tracker_start_time

    if not is_tracking:
        return 

    ret, frame = cap.read()
    if ret:
        timestamp = time.time()
        frame_timestamp_ms = int(timestamp * 1000)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        gaze_direction = "Unknown"
        yawn_status = "No"
        blink_status = "Open"
        face_detected = 0
        ear = 0.0
        mar = 0.0
        perclos_score = 0.0

        if results.face_landmarks:
            face_detected = 1
            face_landmarks = results.face_landmarks[0]

            # Gaze Tracking Logic
            left_iris = face_landmarks[468]
            left_eye_outer = face_landmarks[33]
            left_eye_inner = face_landmarks[133]

            h, w, _ = frame.shape
            iris_x = int(left_iris.x * w)
            eye_outer_x = int(left_eye_outer.x * w)
            eye_inner_x = int(left_eye_inner.x * w)

            eye_width = eye_inner_x - eye_outer_x
            
            if eye_width != 0: 
                iris_position = (iris_x - eye_outer_x) / eye_width
                if iris_position < 0.4:
                    gaze_direction = "Right"
                elif iris_position > 0.6:
                    gaze_direction = "Left"
                else:
                    gaze_direction = "Center"

            # EAR(Eye Aspect Ratio) Logic
            p159, p145 = face_landmarks[159], face_landmarks[145]
            p158, p144 = face_landmarks[158], face_landmarks[144]
            p33, p133 = face_landmarks[33], face_landmarks[133]

            dist_v1 = ((p159.x - p145.x)**2 + (p159.y - p145.y)**2)**0.5
            dist_v2 = ((p158.x - p144.x)**2 + (p158.y - p144.y)**2)**0.5
            dist_h = ((p33.x - p133.x)**2 + (p33.y - p133.y)**2)**0.5

            if dist_h > 0:
                ear = (dist_v1 + dist_v2) / (2.0 * dist_h)

            # Determine if eye is closed for this specific frame
            is_closed = 1 if ear < 0.25 else 0
            
            if is_closed:
                blink_status = "Closed"

            # PERCLOS Logic (Dinges & Grace, 1998 / Gao et al., 2014)
            # 1. Append current state with a timestamp
            perclos_buffer.append((timestamp, is_closed))

            # 2. Remove any frames older than 60 seconds
            while len(perclos_buffer) > 0 and (timestamp - perclos_buffer[0][0]) > 60.0:
                perclos_buffer.pop(0)

            # 3. Calculate the percentage of closed frames in the 60s window
            if len(perclos_buffer) > 0:
                closed_frames = sum(state for _, state in perclos_buffer)
                perclos_score = closed_frames / len(perclos_buffer)

            # 4. Trigger fatigue warning if threshold > 15% AND 60s grace period is over
            elapsed_time = timestamp - tracker_start_time
            if elapsed_time > 60.0 and perclos_score > 0.15:
                blink_status = "FATIGUE (PERCLOS)"

            # MAR(Mouth Aspect Ratio) Logic
            p13, p14 = face_landmarks[13], face_landmarks[14]
            p61, p291 = face_landmarks[61], face_landmarks[291]

            dist_mouth_v = ((p13.x - p14.x)**2 + (p13.y - p14.y)**2)**0.5
            dist_mouth_h = ((p61.x - p291.x)**2 + (p61.y - p291.y)**2)**0.5

            if dist_mouth_h > 0:
                mar = dist_mouth_v / dist_mouth_h
            
            # Threshold for yawning
            if mar > 0.5: 
                if yawn_start_time is None:
                    yawn_start_time = timestamp 
                
                if (timestamp - yawn_start_time) > 4:
                    yawn_status = "YAWNING"
            else:
                yawn_start_time = None

            # Drawing Visuals
            cv2.circle(frame, (iris_x, int(left_iris.y * h)), 3, (0, 255, 0), -1)
            
            if yawn_status == "YAWNING":
                cv2.putText(frame, "YAWN DETECTED", (30, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
            if blink_status == "Closed":
                cv2.putText(frame, "EYES CLOSED", (30, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            elif blink_status == "FATIGUE (PERCLOS)":
                cv2.putText(frame, "FATIGUE WARNING!", (30, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Display PERCLOS score
            if elapsed_time <= 60.0:
                time_left = int(60.0 - elapsed_time)
                cv2.putText(frame, f"PERCLOS: Calibrating ({time_left}s)", 
                            (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"PERCLOS: {perclos_score*100:.1f}%", 
                            (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Log to CSV
        csv_writer.writerow([timestamp, gaze_direction, yawn_status, blink_status, face_detected, round(ear, 3), round(mar, 3), round(perclos_score, 3)])

        # Display gaze direction
        cv2.putText(frame, f"Gaze: {gaze_direction}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Tkinter Video Embedding
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Loop frame
    root.after(10, process_frame)

def stop_tracker():
    global is_tracking, cap, landmarker, log_file
    is_tracking = False
    
    if cap: cap.release()
    if landmarker: landmarker.close()
    if log_file: log_file.close()
    
    video_label.configure(image='')

### Tkinter interface ###
root = tk.Tk()
root.title("Gaze and Engagement Tracker")
root.geometry("800x600") 

label = tk.Label(root, text="Driver Gaze and Engagement Tracker", font=("Arial", 16, "bold"))
label.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

start_button = tk.Button(btn_frame, text="Start Tracker", command=start_tracker, bg="green", fg="white", width=15)
start_button.grid(row=0, column=0, padx=10)

stop_button = tk.Button(btn_frame, text="Stop Tracker", command=stop_tracker, bg="red", fg="white", width=15)
stop_button.grid(row=0, column=1, padx=10)

video_label = tk.Label(root)
video_label.pack(pady=10)

root.mainloop()