import cv2

import time
import os
import csv
import winsound  
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection

# Initialize webcamqqqq
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Log directories
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# CSV log
csv_log_file = "cheating_log.csv"
if not os.path.exists(csv_log_file):
    with open(csv_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Type", "Details"])

# Calibration setup
calibrated_angles = None
start_time = time.time()
calibration_done = False

# Timers
head_misalignment_start_time = None
eye_misalignment_start_time = None
mobile_detection_start_time = None

# Default values
head_direction = "Looking at Screen"
gaze_direction = "Looking Center"


warning_active = False
warning_start_time = 0
WARNING_DURATION = 5  


def display_warning(frame, warning_text):
    overlay = frame.copy()
    height, width = frame.shape[:2]
    cv2.rectangle(overlay, (30, 30), (width - 30, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, warning_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(frame, "Cheating Detected!", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("Combined Detection", frame)

def play_alert_sound():
    try:
        if os.path.exists("beep.wav"):
            winsound.PlaySound("beep.wav", winsound.SND_FILENAME)
        else:
            winsound.Beep(1000, 500)  # frequency, duration in ms
    except Exception as e:
        print(f"Sound error: {e}")

def log_cheating_event(event_type, details):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(csv_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event_type, details])



while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Eye detection
    try:
        frame, gaze_direction = process_eye_movement(frame)
        cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        print(f"Eye error: {e}")
        gaze_direction = "Error"

    # Head pose
    try:
        if not calibration_done:
            cv2.putText(frame, "Calibrating... Keep your head straight", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if time.time() - start_time <= 5:
                result = process_head_pose(frame, None)
                if result and len(result) == 2:
                    _, calibrated_angles = result
            else:
                calibration_done = True
                if calibrated_angles is None:
                    print("Using default calibration.")
                    calibrated_angles = (0, 0, 0)
        else:
            result = process_head_pose(frame, calibrated_angles)
            if result and len(result) == 2:
                frame, head_direction = result
                cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        print(f"Head error: {e}")
        head_direction = "Error"

    # Mobile detection
    try:
        frame, mobile_detected = process_mobile_detection(frame)
        cv2.putText(frame, f"Mobile Detected: {mobile_detected}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        print(f"Mobile error: {e}")
        mobile_detected = False

    # Check cheating
    cheating_detected = (
        head_direction != "Looking at Screen" and head_direction != "Error" or
        gaze_direction != "Looking Center" and gaze_direction != "Error" or 
        mobile_detected
    )

    # Warnings
    if cheating_detected:
        if not warning_active:
            warning_active = True
            warning_start_time = time.time()
        elif time.time() - warning_start_time < WARNING_DURATION:
            display_warning(frame, "Warning: Cheating Detected!")
            play_alert_sound()
            if head_direction != "Looking at Screen" and head_direction != "Error":
                log_cheating_event("Head Misalignment", head_direction)
            if gaze_direction != "Looking Center" and gaze_direction != "Error":
                log_cheating_event("Eye Movement", gaze_direction)
            if mobile_detected:
                log_cheating_event("Mobile Detection", "Phone Detected")
        else:
            warning_active = False
    else:
        warning_active = False

    # Logging screenshots
    if head_direction != "Looking at Screen" and head_direction != "Error":
        if head_misalignment_start_time is None:
            head_misalignment_start_time = time.time()
        elif time.time() - head_misalignment_start_time >= 3:
            filename = os.path.join(log_dir, f"head_{head_direction}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            head_misalignment_start_time = None
    else:
        head_misalignment_start_time = None

    if gaze_direction != "Looking Center" and gaze_direction != "Error":
        if eye_misalignment_start_time is None:
            eye_misalignment_start_time = time.time()
        elif time.time() - eye_misalignment_start_time >= 3:
            filename = os.path.join(log_dir, f"eye_{gaze_direction}_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            eye_misalignment_start_time = None
    else:
        eye_misalignment_start_time = None

    if mobile_detected:
        if mobile_detection_start_time is None:
            mobile_detection_start_time = time.time()
        elif time.time() - mobile_detection_start_time >= 3:
            filename = os.path.join(log_dir, f"mobile_detected_{int(time.time())}.png")
            cv2.imwrite(filename, frame)
            mobile_detection_start_time = None
    else:
        mobile_detection_start_time = None

    # Display
    if not warning_active:
        cv2.imshow("Combined Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

