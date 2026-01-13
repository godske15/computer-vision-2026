import numpy as np
import cv2

cap = cv2.VideoCapture(cv2.samples.findFile("../images/MVI_2469.MOV"))

# 16:9 med bredde 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 608)

ret, frame1 = cap.read()
# Resize hvis videoen stadig er stor
if frame1.shape[1] > 1080:
    frame1 = cv2.resize(frame1, (1080, 608))

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    # Resize frame
    if frame2.shape[1] > 1080:
        frame2 = cv2.resize(frame2, (1080, 608))
    
    # ================= HSV MASKER =================
    hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    
    # Gul
    lower_yellow = np.array([20, 10, 10])
    upper_yellow = np.array([40, 255, 200])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Blå
    lower_blue = np.array([100, 20, 20])
    upper_blue = np.array([120, 255, 170])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Grøn
    lower_green = np.array([60, 6, 42])
    upper_green = np.array([85, 50, 140])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Pink
    lower_pink = np.array([170, 50, 50])
    upper_pink = np.array([180, 255, 255])
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Kombiner alle masker
    mask = cv2.bitwise_or(mask_yellow, mask_blue)
    mask = cv2.bitwise_or(mask, mask_green)
    mask = cv2.bitwise_or(mask, mask_pink)
    
    # Fjern støj med morphology
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 2)
    
    # ================= CANNY EDGE DETECTION =================
    canny = cv2.Canny(image=mask, threshold1=100, threshold2=200)
    
    # ================= OPTICAL FLOW =================
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 
                                        pyr_scale=0.5,
                                        levels=3,
                                        winsize=15,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # ================= FIND PUNKTER PÅ CANNY EDGES =================
    # Find alle edge pixels
    edge_points = np.column_stack(np.where(canny > 0))
    
    vis = frame2.copy()
    
    # Definer område at ignorere (top og bund)
    h = frame2.shape[0]
    ignore_top = int(h * 0.25)     # Ignorer top 25% af billedet
    ignore_bottom = int(h * 0.70)  # Ignorer bund 30% af billedet
    
    # Subsample edge points for bedre performance
    step = 20  # Øget fra 5 til 10 for færre vektorer
    edge_points = edge_points[::step]
    
    # Tegn vektorer på edge points
    for point in edge_points:
        y, x = point
        
        # Spring punkter i top og bund over
        if y < ignore_top or y > ignore_bottom:
            continue
        
        # Hent flow på dette punkt
        fx = flow[y, x, 0]
        fy = flow[y, x, 1]
        flow_mag = mag[y, x]
        
        # Alle vektorer er grønne
        color = (0, 255, 0)
        
        # Tegn vektor
        if flow_mag > 0.5:
            end_x = int(x + fx * 8)
            end_y = int(y + fy * 8)
            cv2.arrowedLine(vis, (x, y), (end_x, end_y), 
                          color, 2, tipLength=0.3)
    
    cv2.imshow('Canny Edges', canny)
    cv2.imshow('Optical Flow Vectors', vis)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalvectors.png', vis)
        cv2.imwrite('canny.png', canny)
        print("Billeder gemt!")

    prvs = next

cv2.destroyAllWindows()