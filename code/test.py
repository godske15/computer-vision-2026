import cv2
import numpy as np

def contourDetection(bgr):
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ================= FIND CONTOURS =================

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrer på areal
    min_area = 50
    max_area = 500
    filtered_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            filtered_contours.append(cnt)

    # ================= TEGN CONTOURS =================

    result = bgr.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    # ================= VIS =================

    #cv2.imshow("Original", bgr)
    cv2.imshow("HSV Mask", mask)
    cv2.imshow("Filtered Contours", result)


# ================= VIDEO LOOP =================

cap = cv2.VideoCapture("../images/MVI_2469.MOV")  # Brug 0 for webcam

if not cap.isOpened():
    print("Kunne ikke åbne video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    contourDetection(frame)

    # Tryk 'q' for at afslutte
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
