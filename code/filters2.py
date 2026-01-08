import cv2  
import numpy as np 
from matplotlib import pyplot as plt 

gray = cv2.imread("../images/disc3.jpg", 0)
bgr = cv2.imread("../images/disc3.jpg")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

gauss = cv2.GaussianBlur(gray, (5, 5), 0)
bilat = cv2.bilateralFilter(gray, 5, sigmaColor=75, sigmaSpace=75)

def compareEdges(filteredImg):
    sobelx = cv2.Sobel(src=filteredImg, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(src=filteredImg, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelxy = cv2.Sobel(src=filteredImg, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    canny = cv2.Canny(image=filteredImg, threshold1=100, threshold2=200)
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(filteredImg, cv2.CV_64F))

    titles = ["Original", "Sobel", "Canny", "Laplace"]
    images = [filteredImg, sobelxy, canny, laplacian]

    #titles = ["Original", "Sobelx", "Sobely", "Sobelxy"]
    #images = [filteredImg, sobelx, sobely, sobelxy]

    for i in range(len(images)):
        plt.subplot(2,2,i+1), plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def compareThresholds(blurred_grayimg):
    th1 = cv2.adaptiveThreshold(blurred_grayimg, 150, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th2 = cv2.adaptiveThreshold(blurred_grayimg, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, th3 = cv2.threshold(blurred_grayimg, 100, 200, cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    titles = ["Original", "Gaussian", "Adaptive", "Otsu"]
    images = [blurred_grayimg, th1, th2, th3]

    for i in range(len(images)):
        plt.subplot(2,2,i+1), plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def hueEdges(hsvimg):
    shift = 25
    h, s, v = cv2.split(hsvimg)
    shiftedHue = h.copy()

    height = shiftedHue.shape[0]
    width = shiftedHue.shape[1]
    for y in range(0, height):
        for x in range(0, width):
            shiftedHue[y, x] = (h[y, x] + shift)%180

    canny = cv2.Canny(shiftedHue, 150, 255)

    cv2.imshow("Canny on shifted hue", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def filterYellowHSV(hsvimg):

    # Lav HSV maske for blå farve
    lower_blue = np.array([90, 0, 0])
    upper_blue = np.array([145, 150, 255])
    
    mask1 = cv2.inRange(hsvimg, lower_blue, upper_blue)

    # Lav HSV maske for grøn farve
    lower_green = np.array([35, 10, 20])
    upper_green = np.array([80, 255, 255])
    
    mask2 = cv2.inRange(hsvimg, lower_green, upper_green)

    mask = np.maximum.reduce([mask1, mask2])
    
    cv2.imshow("Canny on shifted hue", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contourDetection(hsvimg):
    # Lav HSV maske for gul farve
    lower_yellow = np.array([20, 100, 150])
    upper_yellow = np.array([40, 255, 255])
    
    mask1 = cv2.inRange(hsvimg, lower_yellow, upper_yellow)

    # Lav HSV maske for blå farve
    lower_blue = np.array([95, 0, 0])
    upper_blue = np.array([140, 150, 255])
    
    mask2 = cv2.inRange(hsvimg, lower_blue, upper_blue)

    # Lav HSV maske for grøn farve
    lower_green = np.array([35, 30, 100])
    upper_green = np.array([80, 40, 140])
    
    mask3 = cv2.inRange(hsvimg, lower_green, upper_green)

    # Lav HSV maske for pink farve
    lower_pink = np.array([170, 50, 50])
    upper_pink = np.array([180, 255, 255])
    
    mask4 = cv2.inRange(hsvimg, lower_pink, upper_pink)

    # Læg maskerne sammen, således at der bliver kørt contour på alle HSV filtrerne
    mask = np.maximum.reduce([mask1, mask2, mask3, mask4])
    
    # Anvend morphology for at fjerne støj (valgfrit men anbefalet)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours DIREKTE på HSV masken
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrer contours baseret på areal
    filtered_contours = []
    min_area = 100
    max_area = 500
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            filtered_contours.append(contour)
    
    print(f"Fandt {len(contours)} contours, {len(filtered_contours)} efter filtrering")
    
    # Tegn contours på original billede
    result = bgr.copy()
    cv2.drawContours(result, filtered_contours, -1, (0,255,0), 2, cv2.LINE_AA)
    
    cv2.imshow("HSV Mask", mask)
    cv2.imshow("Filtered Contours", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blobDetection(grayimg):
    parameters = cv2.SimpleBlobDetector_Params()

    parameters.filterByArea = True
    parameters.minArea = 100
    parameters.filterByCircularity = True 
    parameters.minCircularity = 0.1
    parameters.maxCircularity = 1.0
    parameters.filterByInertia = True
    parameters.minInertiaRatio = 0.1
    parameters.filterByConvexity = False
    parameters.minConvexity = 0.1

    detector = cv2.SimpleBlobDetector_create(parameters)

    dst = cv2.equalizeHist(grayimg) # Higher contrast in the gray image to detect objects easier

    keypoints = detector.detect(dst)
    imageWithKeypoints = cv2.drawKeypoints(dst, 
                                           keypoints, 
                                           np.array([]), 
                                           (0,0,255), 
                                           cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
                                           )
    cv2.imshow("Blobs", imageWithKeypoints)

    if cv2.waitKey(0) & 0xFF == ord('q'): # Tryk 'q' for at afslutte
        cv2.destroyAllWindows()


def showComparison():
    cv2.imshow("org", bgr)
    cv2.imshow("gauss", gauss)
    cv2.imshow("bilat", bilat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#compareThresholds(bilat)
#blobDetection(gray)
#compareEdges(gauss)
#hueEdges(hsv)
contourDetection(hsv)
#showComparison()
#filterYellowHSV(hsv)