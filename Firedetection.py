import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import subprocess
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    filename='fire_detection_yolo.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("No camera detected")
    print("Error: No camera detected")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Audio system (macOS-specific using 'say')
siren_playing = False
siren_thread = None
audio_enabled = True

def play_audio_async(text):
    try:
        subprocess.Popen(['say', '-v', 'Daniel', text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logging.error(f"Audio playback error: {e}")

def play_siren_continuous():
    global siren_playing, audio_enabled
    while siren_playing and audio_enabled:
        play_audio_async("Fire detected! Evacuate immediately!")
        time.sleep(3.0)

def start_siren():
    global siren_playing, siren_thread, audio_enabled
    if not siren_playing and audio_enabled:
        siren_playing = True
        siren_thread = threading.Thread(target=play_siren_continuous, daemon=True)
        siren_thread.start()
        logging.info("Audio alert initiated")
        print("Alert: Fire detection audio activated")

def stop_siren():
    global siren_playing
    siren_playing = False
    logging.info("Audio alert deactivated")
    print("Alert: Audio deactivated")

# Load YOLO model
try:
    model = YOLO("best.pt")  # Replace with path to your best.pt
    logging.info("Loaded YOLO model: best.pt")
    print("System: Successfully loaded YOLO model")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    print(f"Error: Failed to load model: {e}")
    print("Ensure best.pt is in the script directory or provide the correct path")
    exit()

# Configuration
config = {
    'conf_threshold': 0.7,
    'iou_threshold': 0.5,
    'fire_class_ids': [0],  # Assume 'fire' is class ID 0
    'emergency_threshold': 0.8,
    'detection_history_len': 15
}

# Initialize detection history
confirmed_fire_history = deque(maxlen=config['detection_history_len'])
frame_count = 0
test_mode = False

# Main loop
print("System: Initializing fire detection system...")
print("Controls: 'q'=Exit, 's'=Save Snapshot, 'm'=Toggle Audio, 'r'=Reset, 't'=Test Mode")
print("-" * 70)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        logging.error("Camera read error")
        print("Error: Camera read error")
        break
    
    frame_count += 1
    annotated = frame.copy()
    
    # Draw semi-transparent status bar
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (1280, 80), (50, 50, 50), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
    
    if test_mode:
        # Test mode: show raw camera feed with brightness info
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        cv2.putText(annotated, f"Test Mode - Average Brightness: {avg_brightness:.1f}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, "Press 't' to exit Test Mode", (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        try:
            # Run YOLO inference
            results = model.predict(frame, conf=config['conf_threshold'], iou=config['iou_threshold'], classes=config['fire_class_ids'])
            fire_detected = False
            detection_count = 0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    if cls in config['fire_class_ids']:
                        fire_detected = True
                        detection_count += 1
                        # Draw professional bounding box and label
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 200), 3)
                        cv2.rectangle(annotated, (x1, y1-60), (x1+300, y1), (0, 0, 200), -1)
                        cv2.putText(annotated, f"Fire Detected: {conf:.2f} Confidence", (x1+5, y1-35), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(annotated, "Immediate Action Required", (x1+5, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 200), 1)
                        logging.info(f"Fire detected: confidence={conf:.2f}, bbox=({x1},{y1},{x2},{y2})")
                        print(f"Alert: Fire detected with {conf:.2f} confidence")
                    else:
                        # Non-fire detection
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 200), 1)
                        cv2.putText(annotated, f"Non-Fire: {class_name} ({conf:.2f})", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        logging.debug(f"Non-fire detection: class={class_name}, confidence={conf:.2f}")
            
            confirmed_fire_history.append(fire_detected)
            
            # Check for emergency
            if len(confirmed_fire_history) >= 8:
                recent = list(confirmed_fire_history)[-8:]
                confirmation_ratio = sum(recent) / len(recent)
                fire_emergency = confirmation_ratio >= config['emergency_threshold']
            else:
                confirmation_ratio = 0.0
                fire_emergency = False
            
            if fire_emergency and not siren_playing:
                start_siren()
            elif not fire_emergency and siren_playing:
                stop_siren()
            
            # Display professional status
            if fire_emergency:
                status_color = (0, 0, 200)
                status_text = "Critical Alert: Confirmed Fire Emergency"
                if int(time.time() * 3) % 2:
                    status_color = (255, 255, 255)
            elif fire_detected:
                status_color = (0, 100, 200)
                status_text = "Warning: Fire Detected - Confirming"
            elif detection_count > 0:
                status_color = (0, 200, 200)
                status_text = f"Potential Detection: {detection_count} Objects (Insufficient Evidence)"
            else:
                status_color = (0, 200, 0)
                status_text = "Secure: No Fire Detected"
            
            cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            cv2.putText(annotated, f"Confidence Level: {int(confirmation_ratio*100)}% | Frame: {frame_count}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            fps = 1 / (time.time() - start_time)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (1150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(annotated, "Fire Detection System - YOLOv8", (10, 710), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        except Exception as e:
            logging.error(f"Inference error: {e}")
            print(f"Error: Inference failed: {e}")
    
    cv2.imshow("Fire Detection System", annotated)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("System: Shutting down fire detection")
        break
    elif key == ord('s'):
        filename = f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, annotated)
        logging.info(f"Snapshot saved: {filename}")
        print(f"System: Snapshot saved as {filename}")
    elif key == ord('m'):
        audio_enabled = not audio_enabled
        if not audio_enabled:
            stop_siren()
        logging.info(f"Audio {'enabled' if audio_enabled else 'disabled'}")
        print(f"System: Audio {'enabled' if audio_enabled else 'disabled'}")
    elif key == ord('r'):
        confirmed_fire_history.clear()
        stop_siren()
        logging.info("System reset")
        print("System: Detection system reset")
    elif key == ord('t'):
        test_mode = not test_mode
        logging.info(f"Test mode {'enabled' if test_mode else 'disabled'}")
        print(f"System: Test mode {'enabled' if test_mode else 'disabled'}")

# Cleanup
stop_siren()
cap.release()
cv2.destroyAllWindows()
try:
    subprocess.run(['killall', 'say'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    pass
logging.info("Fire detection system shutdown")
print("System: Fire detection system shutdown complete")