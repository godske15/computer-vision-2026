import numpy as np
import cv2

cap = cv2.VideoCapture(cv2.samples.findFile("../images/MVI_2469.MOV"))

# 16:9 med bredde 680
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)

ret, frame1 = cap.read()
if not ret:
    print("Kunne ikke læse video")
    exit()

if frame1.shape[1] > 680:
    frame1 = cv2.resize(frame1, (680, 420))

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# ================= ROI (LILLE KASSE – KURV) =================
ROI_X = 95
ROI_Y = 170
ROI_W = 65
ROI_H = 70

# ================= ROI (STOR KASSE – OMRÅDE) =================
ROI2_X = 30
ROI2_Y = 100
ROI2_W = 195
ROI2_H = 210

# ================= SCORE STATE =================
score = 0
scored = False

# ================= SCORE CONFIRMATION LOGIK =================
score_pending = False
pending_frames = 0
PENDING_CONFIRM_FRAMES = 5

while True:
    ret, frame2 = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    if frame2.shape[1] > 680:
        frame2 = cv2.resize(frame2, (680, 420))

    # ================= HSV MASKER =================
    hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    masks = [
        cv2.inRange(hsv, np.array([20, 10, 10]), np.array([40, 255, 200])),   # gul
        cv2.inRange(hsv, np.array([100, 20, 20]), np.array([120, 255, 170])), # blå
        cv2.inRange(hsv, np.array([60, 6, 42]), np.array([85, 50, 140])),     # grøn
        cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))  # pink
    ]

    for m in masks:
        mask = cv2.bitwise_or(mask, m)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ================= CANNY =================
    canny = cv2.Canny(mask, 100, 200)

    # ================= OPTICAL FLOW =================
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prvs, next_gray, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # ================= EDGE POINTS =================
    edge_points = np.column_stack(np.where(canny > 0))[::20]

    vis = frame2.copy()

    # ================= FLOW VEKTORER (kun i lille ROI) =================
    for y, x in edge_points:
        if ROI2_X < x < ROI2_X + ROI2_W and ROI2_Y < y < ROI2_Y + ROI2_H:
            if mag[y, x] > 0.5:
                fx, fy = flow[y, x]
                cv2.arrowedLine(
                    vis,
                    (x, y),
                    (int(x + fx * 8), int(y + fy * 8)),
                    (0, 255, 0),
                    2,
                    tipLength=0.3
                )

    # ================= ROI-BEVÆGELSE =================
    moving_small = 0
    moving_large = 0

    for y, x in edge_points:
        if mag[y, x] > 0.8:
            if ROI_X < x < ROI_X + ROI_W and ROI_Y < y < ROI_Y + ROI_H:
                moving_small += 1
            if ROI2_X < x < ROI2_X + ROI2_W and ROI2_Y < y < ROI2_Y + ROI2_H:
                moving_large += 1

    SMALL_MOVING = moving_small > 5
    LARGE_MOVING = moving_large > 5

    # ================= SCORE LOGIK =================

    # Start score-kandidat
    if SMALL_MOVING and not score_pending and not scored:
        score_pending = True
        pending_frames = 0
        print("Score candidate")

    # Overvåg de næste frames
    if score_pending:
        pending_frames += 1

        # Disc falder ud
        if LARGE_MOVING and not SMALL_MOVING:
            score_pending = False
            print("Cancelled (fell out)")

        # Score bekræftet
        elif pending_frames >= PENDING_CONFIRM_FRAMES:
            score += 1
            scored = True
            score_pending = False
            print("SCORE!")

    # Reset når alt er stille
    if not SMALL_MOVING and not LARGE_MOVING:
        scored = False

    # ================= VISUALS =================
    cv2.rectangle(vis, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 0, 255), 2)
    cv2.rectangle(vis, (ROI2_X, ROI2_Y), (ROI2_X + ROI2_W, ROI2_Y + ROI2_H), (255, 0, 0), 2)

    if score_pending:
        cv2.putText(vis, "CHECKING",
                    (ROI_X, ROI_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

    cv2.putText(vis, f"Score: {score}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    cv2.imshow("Optical Flow + Disc Golf Scoring", vis)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    prvs = next_gray

cv2.destroyAllWindows()
