import numpy as np
import cv2
import sys

def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # Read the video and first frame
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Kunne ikke åbne video")
        sys.exit()

    # Læs første frame
    ret, frame = cap.read()
    if not ret:
        print("Kunne ikke læse video")
        sys.exit()

    # Create HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        if not ret:
            break

        frame_copy = new_frame.copy()  # Make a copy of the frame for display

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:  # Exit on ESC key
            break

        # Update the previous frame
        old_frame = new_frame

    cap.release()
    cv2.destroyAllWindows()

# Call the function with the correct optical flow method
dense_optical_flow(cv2.calcOpticalFlowFarneback, ".../images/MVI_2469.MOV", [None], True)