import numpy as np
import cv2

# Aktiver OpenCL
cv2.ocl.setUseOpenCL(True)

# Check om det virker
print(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")
print(f"OpenCL device: {cv2.ocl.Device.getDefault().name()}")

cap = cv2.VideoCapture(cv2.samples.findFile("../images/MVI_2469.MOV"))

# ================= WINDOW SIZE SETTINGS =================
WINDOW_WIDTH = 1020  # Ændr denne til ønsket bredde (dobbelt af 680)
WINDOW_HEIGHT = 630  # Ændr denne til ønsket højde (dobbelt af 420)

# Original størrelse (reference)
ORIGINAL_WIDTH = 680
ORIGINAL_HEIGHT = 420

# Beregn skalerings-faktor
SCALE_X = WINDOW_WIDTH / ORIGINAL_WIDTH
SCALE_Y = WINDOW_HEIGHT / ORIGINAL_HEIGHT

ret, frame1 = cap.read()
if not ret:
    print("Kunne ikke læse video")
    exit()

# Resize til den ønskede vinduesstørrelse
frame1 = cv2.resize(frame1, (WINDOW_WIDTH, WINDOW_HEIGHT))

# Konverter første frame til GPU
frame_gpu = cv2.UMat(frame1)
prvs = cv2.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)

# ================= ROI (LILLE KASSE – KURV) - Original værdier =================
# Disse værdier er fra den originale 680x420 størrelse
ORIG_ROI_X = 95
ORIG_ROI_Y = 170
ORIG_ROI_W = 65
ORIG_ROI_H = 70

# ================= ROI (STOR KASSE – OMRÅDE) - Original værdier =================
ORIG_ROI2_X = 30
ORIG_ROI2_Y = 100
ORIG_ROI2_W = 195
ORIG_ROI2_H = 210

# ================= SKALEREDE ROI VÆRDIER =================
ROI_X = int(ORIG_ROI_X * SCALE_X)
ROI_Y = int(ORIG_ROI_Y * SCALE_Y)
ROI_W = int(ORIG_ROI_W * SCALE_X)
ROI_H = int(ORIG_ROI_H * SCALE_Y)

ROI2_X = int(ORIG_ROI2_X * SCALE_X)
ROI2_Y = int(ORIG_ROI2_Y * SCALE_Y)
ROI2_W = int(ORIG_ROI2_W * SCALE_X)
ROI2_H = int(ORIG_ROI2_H * SCALE_Y)

# ================= SCORE STATE =================
score = 0
scored = False

# ================= SCORE CONFIRMATION LOGIK =================
score_pending = False
pending_frames = 0
PENDING_CONFIRM_FRAMES = 25

while True:
    ret, frame2 = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    # Resize til den ønskede vinduesstørrelse
    frame2 = cv2.resize(frame2, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Konverter frame til GPU
    frame2_gpu = cv2.UMat(frame2)

    # ================= HSV MASKER (på GPU) =================
    hsv = cv2.cvtColor(frame2_gpu, cv2.COLOR_BGR2HSV)

    # Start med første mask
    mask = cv2.inRange(hsv, np.array([20, 40, 20]), np.array([40, 255, 200]))  # gul
    
    # Tilføj de andre masker
    masks = [
        cv2.inRange(hsv, np.array([100, 10, 10]), np.array([120, 255, 255])), # blå
        cv2.inRange(hsv, np.array([60, 6, 42]), np.array([85, 50, 140])),     # grøn
        cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))  # pink
    ]

    for m in masks:
        mask = cv2.bitwise_or(mask, m)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ================= CANNY (på GPU) =================
    canny = cv2.Canny(mask, 100, 200)

    # ================= OPTICAL FLOW (på GPU) =================
    next_gray = cv2.cvtColor(frame2_gpu, cv2.COLOR_BGR2GRAY)

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

    # Split flow channels (GPU-kompatibel måde)
    flow_x, flow_y = cv2.split(flow)
    
    # Beregn magnitude (på GPU)
    mag = cv2.magnitude(flow_x, flow_y)

    # Konverter kun det nødvendige til CPU for videre processing
    canny_cpu = canny.get()
    mag_cpu = mag.get()
    flow_cpu = flow.get()

    # ================= EDGE POINTS =================
    edge_points = np.column_stack(np.where(canny_cpu > 0))[::15]

    vis = frame2.copy()

    # ================= FLOW VEKTORER (kun i lille Kasserne) =================
    for y, x in edge_points:
        if ROI2_X < x < ROI2_X + ROI2_W and ROI2_Y < y < ROI2_Y + ROI2_H:
            if mag_cpu[y, x] > 0.5:
                fx, fy = flow_cpu[y, x]
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
        if mag_cpu[y, x] > 0.8:
            if ROI_X < x < ROI_X + ROI_W and ROI_Y < y < ROI_Y + ROI_H:
                moving_small += 1
            if ROI2_X < x < ROI2_X + ROI2_W and ROI2_Y < y < ROI2_Y + ROI2_H:
                moving_large += 1

    SMALL_MOVING = moving_small > 2
    LARGE_MOVING = moving_large > 2

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