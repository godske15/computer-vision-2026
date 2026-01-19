import cv2
import sys
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def create_tracker(tracker_type):
    """Opretter en tracker baseret på type"""
    if int(minor_ver) < 3:
        return cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            return cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            return cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            return cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()

if __name__ == '__main__':
    # Vælg tracker type
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]  # KCF
    
    # Liste til at holde styr på alle trackere og deres bounding boxes
    trackers = []
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    # Read video
    video = cv2.VideoCapture("../images/MVI_2469.MOV")
    
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()
    
    # Vælg første bounding box
    print("Vælg første objekt at tracke (tryk ENTER eller SPACE for at bekræfte, C for at annullere)")
    bbox = cv2.selectROI("Tracking", frame, False)
    
    if bbox != (0, 0, 0, 0):
        tracker = create_tracker(tracker_type)
        tracker.init(frame, bbox)
        trackers.append({'tracker': tracker, 'bbox': bbox, 'color': colors[0]})
    
    paused = False
    
    print("\nTaster:")
    print("  SPACE - Pause/fortsæt video")
    print("  N - Tilføj ny tracking-boks (kun når pauset)")
    print("  ESC - Afslut")
    
    while True:
        if not paused:
            # Læs ny frame
            ok, frame = video.read()
            if not ok:
                print("Video slut")
                break
            
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
        
        # Display tracker info
        cv2.putText(frame, tracker_type + " Tracker", (100,20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        cv2.putText(frame, f"FPS: {int(fps)}" if not paused else "PAUSED", (100,50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        cv2.putText(frame, f"Objekter: {len(trackers)}", (100,80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        
        # Vis frame
        cv2.imshow("Tracking", frame)
        
        # Håndter tastatur input
        k = cv2.waitKey(1) & 0xff
        
        if k == 27:  # ESC
            break
        elif k == 32:  # SPACE
            paused = not paused
            if paused:
                print("Video pauset - tryk N for at tilføje ny tracking-boks, SPACE for at fortsætte")
            else:
                print("Video fortsætter")
        elif k == ord('n') or k == ord('N'):
            if paused:
                print("Vælg nyt objekt at tracke")
                bbox = cv2.selectROI("Tracking", frame, False)
                
                if bbox != (0, 0, 0, 0):
                    new_tracker = create_tracker(tracker_type)
                    new_tracker.init(frame, bbox)
                    color = colors[len(trackers) % len(colors)]
                    trackers.append({'tracker': new_tracker, 'bbox': bbox, 'color': color})
                    print(f"Tilføjet objekt {len(trackers)}")
            else:
                print("Sæt videoen på pause først (tryk SPACE)")
    
    video.release()
    cv2.destroyAllWindows()