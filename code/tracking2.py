import cv2
import sys
import numpy as np

# Prøv at importere legacy trackers
try:
    from cv2 import legacy
    TRACKERS_AVAILABLE = True
except ImportError:
    TRACKERS_AVAILABLE = False
    print("Warning: cv2.legacy ikke tilgængelig - installér opencv-contrib-python")

# Aktiver OpenCL
cv2.ocl.setUseOpenCL(True)
print(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")
print(f"OpenCL device: {cv2.ocl.Device.getDefault().name()}")

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# ================= WINDOW SIZE SETTINGS =================
WINDOW_WIDTH = 1020
WINDOW_HEIGHT = 630
ORIGINAL_WIDTH = 680
ORIGINAL_HEIGHT = 420
SCALE_X = WINDOW_WIDTH / ORIGINAL_WIDTH
SCALE_Y = WINDOW_HEIGHT / ORIGINAL_HEIGHT

# ================= HSV RANGES (juster efter dine objekter) =================
HSV_RANGES = {
    'gul': ([20, 40, 20], [40, 255, 200]),
    'blå': ([100, 0, 0], [120, 255, 255]),
    'grøn': ([60, 6, 42], [85, 50, 140]),
    'pink': ([170, 50, 50], [180, 255, 255])
}

def create_tracker(tracker_type):
    """Opretter en tracker baseret på type"""
    if int(minor_ver) < 3:
        return cv2.Tracker_create(tracker_type)
    else:
        # Prøv først legacy trackers
        if TRACKERS_AVAILABLE:
            trackers_dict = {
                'BOOSTING': legacy.TrackerBoosting_create,
                'MIL': legacy.TrackerMIL_create,
                'KCF': legacy.TrackerKCF_create,
                'TLD': legacy.TrackerTLD_create,
                'MEDIANFLOW': legacy.TrackerMedianFlow_create,
                'MOSSE': legacy.TrackerMOSSE_create,
                'CSRT': legacy.TrackerCSRT_create
            }
            if tracker_type in trackers_dict:
                return trackers_dict[tracker_type]()
        
        # Fallback til nye trackers (kun CSRT og KCF tilgængelige)
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        else:
            # Default til CSRT hvis legacy ikke findes
            print(f"Tracker {tracker_type} ikke tilgængelig, bruger CSRT")
            return cv2.TrackerCSRT_create()

def detect_objects_hsv(frame_gpu):
    """Detekterer objekter baseret på HSV-filtrering og Canny (på GPU)"""
    # Konverter til HSV på GPU
    hsv = cv2.cvtColor(frame_gpu, cv2.COLOR_BGR2HSV)
    
    # Start med første mask
    combined_mask = None
    
    for color_name, (lower, upper) in HSV_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Morphological operations på GPU
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Canny edge detection på GPU (ligesom optical flow koden)
    canny = cv2.Canny(combined_mask, 100, 200)
    
    return canny

def find_objects_from_canny(canny_gpu):
    """Finder objekter fra Canny edges og returnerer bounding boxes"""
    # Konverter til CPU
    canny_cpu = canny_gpu.get()
    
    # Find edge points (ligesom i optical flow koden)
    edge_points = np.column_stack(np.where(canny_cpu > 0))
    
    if len(edge_points) == 0:
        return []
    
    # Cluster edge points til at finde separate objekter
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import fcluster, linkage
    
    if len(edge_points) < 2:
        return []
    
    # Beregn afstande mellem edge points
    distances = pdist(edge_points)
    
    # Lav hierarchical clustering
    linkage_matrix = linkage(distances, method='single')
    
    # Find clusters (objekter) med max afstand mellem points
    clusters = fcluster(linkage_matrix, t=50, criterion='distance')
    
    bboxes = []
    for cluster_id in np.unique(clusters):
        cluster_points = edge_points[clusters == cluster_id]
        
        # Beregn bounding box for dette cluster
        if len(cluster_points) > 20:  # Minimum antal points for at være et objekt
            y_coords = cluster_points[:, 0]
            x_coords = cluster_points[:, 1]
            
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            w = x_max - x_min
            h = y_max - y_min
            
            # Kun tilføj hvis det er stort nok
            if w > 10 and h > 10:
                # Tilføj lidt padding
                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                w = w + 2 * padding
                h = h + 2 * padding
                
                bboxes.append((int(x_min), int(y_min), int(w), int(h)))
    
    return bboxes

if __name__ == '__main__':
    # Vælg tracker type - kun CSRT er tilgængelig i nyere OpenCV uden contrib
    if TRACKERS_AVAILABLE:
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
        tracker_type = tracker_types[6]  # CSRT - mest præcis
    else:
        tracker_type = 'CSRT'  # Eneste tilgængelige
        print("Kun CSRT tracker tilgængelig. Installér opencv-contrib-python for flere.")
    
    # Liste til at holde styr på alle trackere
    trackers = []
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    # Read video
    cap = cv2.VideoCapture("../images/MVI_2469.MOV")
    
    if not cap.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot read video file")
        sys.exit()
    
    # Resize til den ønskede vinduesstørrelse
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Konverter til GPU
    frame_gpu = cv2.UMat(frame)
    
    paused = False
    show_hsv_detection = False
    auto_detect_mode = False
    
    print("\nTaster:")
    print("  SPACE - Pause/fortsæt video")
    print("  N - Tilføj ny tracking-boks manuelt (kun når pauset)")
    print("  A - Auto-detect objekter baseret på HSV (kun når pauset)")
    print("  H - Vis/skjul HSV detection overlay")
    print("  C - Clear alle trackere")
    print("  ESC - Afslut")
    
    while True:
        if not paused:
            # Læs ny frame
            ret, frame = cap.read()
            if not ret:
                print("Video slut")
                break
            
            # Resize og konverter til GPU
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame_gpu = cv2.UMat(frame)
            
            # Start timer
            timer = cv2.getTickCount()
            
            # Opdater alle trackere
            for i, tracker_info in enumerate(trackers):
                ok, bbox = tracker_info['tracker'].update(frame)
                tracker_info['bbox'] = bbox
                
                # Tegn bounding box
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, tracker_info['color'], 2, 1)
                    cv2.putText(frame, f"Obj {i+1}", (int(bbox[0]), int(bbox[1])-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker_info['color'], 2)
            
            # Beregn FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Vis HSV detection overlay hvis aktiveret
        if show_hsv_detection:
            canny = detect_objects_hsv(frame_gpu)
            canny_cpu = canny.get()
            
            # Lav overlay med edges (grøn)
            overlay = cv2.cvtColor(canny_cpu, cv2.COLOR_GRAY2BGR)
            overlay[:,:,0] = 0  # Fjern blå kanal
            overlay[:,:,1] = canny_cpu  # Grøn kanal
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Display tracker info
        cv2.putText(frame, tracker_type + " Tracker", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,170,50), 2)
        cv2.putText(frame, f"FPS: {int(fps)}" if not paused else "PAUSED", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,170,50), 2)
        cv2.putText(frame, f"Objekter: {len(trackers)}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,170,50), 2)
        
        if show_hsv_detection:
            cv2.putText(frame, "HSV Detection: ON", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # Vis frame
        cv2.imshow("HSV Multi-Object Tracker", frame)
        
        # Håndter tastatur input
        k = cv2.waitKey(1) & 0xff
        
        if k == 27:  # ESC
            break
        elif k == 32:  # SPACE
            paused = not paused
            if paused:
                print("Video pauset - tryk N for manuel tracking, A for auto-detect")
            else:
                print("Video fortsætter")
        elif k == ord('n') or k == ord('N'):
            if paused:
                print("Vælg nyt objekt at tracke")
                bbox = cv2.selectROI("HSV Multi-Object Tracker", frame, False)
                
                if bbox != (0, 0, 0, 0):
                    new_tracker = create_tracker(tracker_type)
                    new_tracker.init(frame, bbox)
                    color = colors[len(trackers) % len(colors)]
                    trackers.append({'tracker': new_tracker, 'bbox': bbox, 'color': color})
                    print(f"Tilføjet objekt {len(trackers)}")
            else:
                print("Sæt videoen på pause først (tryk SPACE)")
        elif k == ord('a') or k == ord('A'):
            if paused:
                print("Auto-detekterer objekter baseret på HSV + Canny...")
                canny = detect_objects_hsv(frame_gpu)
                bboxes = find_objects_from_canny(canny)
                
                if bboxes:
                    # Clear eksisterende trackere
                    trackers.clear()
                    
                    # Tilføj alle detekterede objekter
                    for bbox in bboxes:
                        new_tracker = create_tracker(tracker_type)
                        new_tracker.init(frame, bbox)
                        color = colors[len(trackers) % len(colors)]
                        trackers.append({'tracker': new_tracker, 'bbox': bbox, 'color': color})
                    
                    print(f"Fundet og tracker {len(bboxes)} objekter")
                else:
                    print("Ingen objekter fundet - juster HSV ranges")
            else:
                print("Sæt videoen på pause først (tryk SPACE)")
        elif k == ord('h') or k == ord('H'):
            show_hsv_detection = not show_hsv_detection
            print(f"HSV detection overlay: {'ON' if show_hsv_detection else 'OFF'}")
        elif k == ord('c') or k == ord('C'):
            trackers.clear()
            print("Alle trackere fjernet")
    
    cap.release()
    cv2.destroyAllWindows()