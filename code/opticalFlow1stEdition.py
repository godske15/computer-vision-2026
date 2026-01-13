import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv2.VideoCapture(args.image)

#../images/MVI_2469.MOV

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)

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

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)

    # Tryk 'q' for at afslutte
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()