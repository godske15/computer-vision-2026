import cv2
import numpy as np
import sys

def contourDetection(bgr):
    """
    Returnerer bounding boxes for detekterede objekter
    """
    # Konverter til HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # ================= HSV MASKER =================

    # Gul
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 200])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Blå
    lower_blue = np.array([95, 20, 20])
    upper_blue = np.array([140, 150, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Grøn
    lower_green = np.array([60, 6, 42])
    upper_green = np.array([85, 40, 140])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Pink
    lower_pink = np.array([170, 50, 50])
    upper_pink = np.array([180, 255, 255])
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

    # ================= KOMBINER MASKER =================

    mask = cv2.bitwise_or(mask_yellow, mask_blue)
    mask = cv2.bitwise_or(mask, mask_green)
    mask = cv2.bitwise_or(mask, mask_pink)

    # ================= MORPHOLOGY =================

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # ================= FIND CONTOURS =================

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrer på areal og lav bounding boxes
    min_area = 50
    max_area = 300
    bboxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, w, h))

    return bboxes, mask


# ================= VIDEO LOOP =================

cap = cv2.VideoCapture("../images/MVI_2469.MOV")

if not cap.isOpened():
    print("Kunne ikke åbne video")
    sys.exit()

# Læs første frame
ret, frame = cap.read()
if not ret:
    print("Kunne ikke læse video")
    sys.exit()

# Variabler til tracking
trackers = []  # Liste af (tracker, farve) tupler
colors = []    # Farver til hver tracker
detection_interval = 30  # Kør detection hver 30. frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Kør detection med jævne mellemrum for at finde nye objekter
    if frame_count % detection_interval == 0:
        bboxes, mask = contourDetection(frame)
        
        # Initialiser nye trackere for detekterede objekter
        # (kun hvis vi ikke allerede tracker for mange)
        if len(trackers) < 10:  # Max 10 objekter
            new_trackers = []
            for bbox in bboxes:
                tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(frame, bbox)
                if ok:
                    # Generer en tilfældig farve for denne tracker
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    new_trackers.append((tracker, color))
            
            trackers = new_trackers

    # Opdater alle trackere
    active_trackers = []
    for tracker, color in trackers:
        ok, bbox = tracker.update(frame)
        
        if ok:
            # Tegn bounding box
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, color, 2, 1)
            active_trackers.append((tracker, color))
    
    # Behold kun aktive trackere
    trackers = active_trackers

    # Vis info
    cv2.putText(frame, f"CSRT Tracker - Objekter: {len(trackers)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, "Tryk 'r' for at nulstille | 'q' for at afslutte", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Tracking", frame)

    # Tastatur input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Nulstil alle trackere
        trackers = []
        frame_count = 0

cap.release()
cv2.destroyAllWindows()