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

# Liste til at holde trackere
trackers = []
tracker_colors = []
next_color_id = 0

# Farver til trackere (cyklisk gennem disse)
colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), 
          (255, 255, 0), (0, 165, 255), (255, 165, 0)]

frame_count = 0
detect_interval = 10  # Detect oftere - hvert 10. frame

while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    # Resize frame
    if frame2.shape[1] > 1080:
        frame2 = cv2.resize(frame2, (1080, 608))
    
    vis = frame2.copy()
    frame_count += 1
    
    # ================= UPDATE EKSISTERENDE TRACKERE =================
    trackers_to_remove = []
    for i, tracker_info in enumerate(trackers):
        tracker = tracker_info['tracker']
        success, bbox = tracker.update(frame2)
        
        if success:
            # Tegn bounding box
            x, y, w, h = [int(v) for v in bbox]
            
            # Valider at bbox er inden for frame
            if x >= 0 and y >= 0 and x + w < frame2.shape[1] and y + h < frame2.shape[0]:
                color = tracker_colors[i]
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                
                # Tegn ID
                cv2.putText(vis, f"ID: {tracker_info['id']}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                trackers_to_remove.append(i)
        else:
            # Marker tracker til removal hvis den fejler
            trackers_to_remove.append(i)
    
    # Fjern fejlede trackere
    for i in sorted(trackers_to_remove, reverse=True):
        del trackers[i]
        del tracker_colors[i]
    
    # ================= DETECT NYE OBJEKTER (HVERT N FRAMES) =================
    if frame_count % detect_interval == 0:
        # HSV MASKER
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        
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
        
        # Kombiner alle masker
        mask = cv2.bitwise_or(mask_yellow, mask_blue)
        mask = cv2.bitwise_or(mask, mask_green)
        mask = cv2.bitwise_or(mask, mask_pink)
        
        # Fjern støj med morphology
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Definer område at ignorere (top og bund)
        h = frame2.shape[0]
        ignore_top = int(h * 0.15)
        ignore_bottom = int(h * 0.85)
        
        # Sæt top og bund til 0 i masken
        mask[:ignore_top, :] = 0
        mask[ignore_bottom:, :] = 0
        
        # Find contours direkte fra masken
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrer baseret på disc størrelse (10-250 pixels område)
            # Area er omtrent pi * r^2, så for diameter 10-250:
            # Min area: ~78 (diameter 10)
            # Max area: ~49000 (diameter 250)
            if area > 49000:
                continue
            
            # Find bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ekstra validering: aspect ratio (discs er nogenlunde cirkulære)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Tjek om der allerede er en tracker på dette område
            is_duplicate = False
            for tracker_info in trackers:
                tx, ty, tw, th = tracker_info['bbox']
                
                # Beregn overlap/distance
                center_x = x + w // 2
                center_y = y + h // 2
                tracker_center_x = tx + tw // 2
                tracker_center_y = ty + th // 2
                
                distance = np.sqrt((center_x - tracker_center_x)**2 + (center_y - tracker_center_y)**2)
                
                # Hvis centrer er tæt på hinanden, considerer det en duplikat
                if distance < max(w, h, tw, th):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Opret ny KCF tracker
                tracker = cv2.TrackerKCF_create()
                bbox = (x, y, w, h)
                
                # Prøv at initialisere tracker
                try:
                    success = tracker.init(frame2, bbox)
                    if success:
                        trackers.append({
                            'tracker': tracker,
                            'id': next_color_id,
                            'bbox': (x, y, w, h)
                        })
                        tracker_colors.append(colors[next_color_id % len(colors)])
                        print(f"Ny tracker oprettet: ID {next_color_id} ved ({x}, {y}) størrelse: {w}x{h}, area: {area}")
                        next_color_id += 1
                except:
                    print(f"Kunne ikke initialisere tracker ved ({x}, {y})")
        
        # Vis masken for debugging
        cv2.imshow('Mask', mask)
        
        prvs = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Vis info
    cv2.putText(vis, f"Trackere: {len(trackers)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, f"Frame: {frame_count}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Tracking', vis)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('tracking.png', vis)
        print("Billede gemt!")

cv2.destroyAllWindows()