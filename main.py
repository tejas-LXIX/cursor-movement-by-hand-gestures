import cv2
import numpy as np  #used to store image as a two-dimensional array
import math
import time
import pyautogui


#Steps are:
# 1. Take the image and convert it to grayscale. This is done because we don't need colour info for this task, and the threshold function only takes a grayscale image as a parameter.

# 2. The grayscale image is blurred to reduce noise in the image(environmental features are reduced). This helps in thresholding the hand properly instead of focusing on some other environmental object.

# 3. Then the blurred image is thresholded. This will show the hand as white and the rest of the background as black. It is a binary image.
# The Otsu Threshold method calculates the threshold value on its own. A pixel whose colour value is above this is converted to black and the reverse as white (Inverse Binary Threshold).

# 4. Next, points on the border of the black and white regions are stored and lines are drawn connecting all these points. This forms the contours of the hand (using the drawContours function).
    #   Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. i.e Your hand represents a contour.

# 5. Around these contour points, a complex hull is formed (using complexHull function) and any deviation of the hand from the complex hull is stored in convexity defects.

# 6. The points on the fingerTIPS  are retrieved from the ‘defects’ variable. This contains the starting point (‘s’ is fingerTIPS 1), end point (‘e’ is fingerTIPS 2), farthest point (‘f’ the point farthest from both the fingerTIPS which lies BETWEEN the two fingers i.e at the starting point of the fingers. See references to understand which point it is) and the distance of the farthest point from the fingertip.

# 7. The points ‘s’, ‘e’, ‘f’ form the vertices of a triangle. The angle measured is the angle at the vertex "f" of the triangle(the point where the two fingers start from). If the angle between ‘s’ and ‘e’ from ‘f’ is less than 90 degrees (the angle between the two fingers) then it is considered as a defect as the defect count is incremented. This means that another finger was opened. This step takes place only when there is a disturbance in the convex hull.

# 8. Based on the defect count, specific mouse functions are linked. EG: 4 defects causes a left click. 3 defects causes a Scroll down. 2 defects causes a Scroll up. 1 defect moves the cursor.

#refer https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html for contourarea,convexhull etc.
#the space between two fingers is considered as one defect. Therefore,for 5 fingers only 4 defects are possible (1,2,3,4). So,only movement,scrollup,scrolldown and left click are possible. no extra defects left for right click .

pyautogui.FAILSAFE = False  #Keeping this enabled(True) will stop the execution of the program once the cursor reaches the bottom left of the screen.
SCREEN_X, SCREEN_Y = pyautogui.size()   # current screen resolution (width and height). Returns 1920*1080 (1080p) or higher for a modern laptop. So,SCREEN_X becomes 1920, SCREEN_Y becomes 1080
CLICK = CLICK_MESSAGE = MOVEMENT_START = None

capture = cv2.VideoCapture(0)   #0 to select the first camera. 1 to select the second camera and so on...

while capture.isOpened():
    ret, img = capture.read()
    CAMERA_Y, CAMERA_X, channels = img.shape    #CAMERA_X and CAMERA_Y denote the dimensions(height,width) of the video that your camera is taking.
    # img.shape returns vertical resolution first. So,CAMERA_Y becomes 480 in this case,and CAMERA_X becomes 640. 480p

    img = cv2.flip(img, 1)  #Flipping the image horizontally,around the y axis(flipcode of value 1). 0 would flip it vertically.
    # This flipping gives us a video-feed flip like we see in a mirror. Does not affect the functioning. I have included it only to make the video-feed not look weird.
    
    copy_img = img
    grey = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)   #convert the image into grayscale.
    cv2.imshow('Grey',grey)
    value = (35, 35)    #kernel size for the gaussian blur
    blurred = cv2.GaussianBlur(grey, value, 0)      #Blur the image.
    cv2.imshow('Blurred', blurred)
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   #returns two parameters,only one is of use to us (thresh1). But the function is such that you have to declare two variables.
    cv2.imshow('Thresholded', thresh1)
    contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #returns a python list of all the contours in the image.
    # Each row in the contour holds the coordinates of all the points that make up the contour.
    #Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. Here,the contour represents your whole hand.
    # Contours is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    # Basically,findContours returns a list of all the 2-d POINTS ON the contour i.e the points that make up your hand in the image.
    max_area = -1
    # len(contours) would IDEALLY be 1 if only the hand is visible in the frame, and the camera doesn't pick up any harsh edges in the hand.
    # In real life though, environmental objects,faces etc. can creep into the webcam's FOV and be considered as a contour.
    # For this reason, the below loop aims to select the most dominant contour (that has the highest area) out of the existing contours. 
    # This contour is then stored in "ci",upon which further processing is done.
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_area = area
            ci = i  #store the index of the most dominant contour in the variable "ci".
    cnt = contours[ci]  #select the dominant contour and store its co-ordinates in the variable "cnt". Further processing is done on cnt and ci.
    ''' FOR TESTING
    print("Contours are")
    print(contours)     prints a list,where each element has 2 points that represent the x and y coordinates of the point.
    print("Specific contour is")
    print(cnt[1][0])
    '''
    x, y, w, h = cv2.boundingRect(cnt)  #The cv2.boundingRect() function of OpenCV is used to draw an approximate rectangle around the binary image. This function is used mainly to highlight the region of interest after obtaining contours from an image.
    cv2.rectangle(copy_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
    hull = cv2.convexHull(cnt)  #to get the points of the convexhull to draw the convexhull contour below.
    drawing = np.zeros(copy_img.shape, np.uint8)    #Another copy of the copy_img(stored as an array),initialised with all zeroes(black empty photo). The function "shape" returns the shape of an array. The shape is a tuple of integers.
    #These numbers denote the lengths of the corresponding array dimension. In other words: The "shape" of an array is a tuple with each value denoting the number of elements per axis (dimension).
    
    #To draw the contours, cv.drawContours function is used.
    # It can also be used to draw any shape provided you have its boundary points. Its first argument is source image, second argument is the contours which should be passed as a Python list.
    # Third argument is index of contours (useful when drawing individual contour. (To draw all contours, pass -1) and remaining arguments are color, thickness etc. 
    
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)     #this is to draw the contours around the fingers.
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)    #to draw the contours of the convex hull. 
    # These both are done together to display to the user the contours around the fingers along with the overall contour of the convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)  #we dont need to obtain the points here.
    defects = cv2.convexityDefects(cnt, hull) #parameters are the contour and the convexhull
    #returns a numpy.ndarray object,which is stored in defects. It is the output vector of convexity defects. Each row in this represents a defect. Each defect represented as a 4-element(start_index, end_index, farthest_pt_index, fixpt_depth), where indices are 0-based indices in the original contour of the convexity defect beginning, end and the farthest point, and fixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the farthest contour point and the hull.
    #basically,these indices represent the positions of the points (start,end,farthest) on the specific contour.
    count_defects = 0

    """ defects EG: 
    [[[    0     3     1   162]]

    [[    3     9     4   114]]

    [[    9    15    11   142]]] """ #defects[i,0] refers to the ith row. EG: defects[0,0] refers to the zeroth row i.e [0,3,1,162]

    defect_dict = None
    for i in range(defects.shape[0]):   #REFER https://theailearner.com/tag/cv2-convexitydefects/   defects.shape[0] gives you the number of defects(number of rows) that have been obtained. So,this loop iterates through each defect and processes them.
        s, e, f, d = defects[i][0]      # s is start index,e is end index,f is farthest point,d is distance between farthest contour point and the hull(basically the distance between farthest point and the fingertip).
        #cnt contains the coordinates of all the points that make up the line that is the contour. So,we can extract co-ordinates of individual points from it.
        start = tuple(cnt[s][0])    #co-ordinates of the starting point (finger1 ka fingertip) 
        end = tuple(cnt[e][0])      #co-ordinates of the ending point (finger2 ka fingertip)
        far = tuple(cnt[f][0])      #co-ordinates of the farthest point (here, it's the point from where the two fingers originate.jahan pe V banta hai)
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57    #calculate the length of each side,to finally calculate the angle between two fingers.
        if angle <= 90:
            count_defects += 1
        cv2.circle(copy_img, far, 5, [0, 0, 255], -1)
        cv2.line(copy_img, start, end, [0, 255, 0], 2)

        if count_defects == 1 and angle <= 90:  #change both this and the count_defects below to change the defects required for scrolling
            defect_dict = {"x": start[0], "y": start[1]}    #dictionary,where x and y are the keys. start[0] and start[1] are the x and y co=ordinates

    #x,y co-ordinates have 0,0 origin at top left corner of the screen
    if defect_dict is not None:
        coords = defect_dict
        if count_defects == 1:
            x = coords['x']   #get the x and y coordinates from the defect_dict dictionary
            y = coords['y']
            display_x = x
            display_y = y

            if MOVEMENT_START is not None:
                M_START = (x, y)
                x = x - MOVEMENT_START[0]   #store the difference between current x value and the previous x value. This is how much the x value has changed.
                y = y - MOVEMENT_START[1]
                x = x * (SCREEN_X / CAMERA_X)   #without these two,the movement of the cursor becomes very slow.
                y = y * (SCREEN_Y / CAMERA_Y)
                MOVEMENT_START = M_START
                print("X: " + str(x) + " Y: " + str(y))
                pyautogui.moveRel(x, y)     #move mouse relative to its current position. x and y represent the change in x and y position relative to the current position of the mouse.
            else:
                MOVEMENT_START = (x, y)

            cv2.circle(copy_img, (display_x, display_y), 5, [255, 255, 255], 20)
        elif count_defects == 2 and CLICK is None:
            CLICK = time.time()     #get the time at which this action has been done.
            CLICK_MESSAGE = "Scroll Up"
            pyautogui.scroll(70)           
        elif count_defects == 3 and CLICK is None:
            CLICK = time.time()
            CLICK_MESSAGE = "Scroll Down"
            pyautogui.scroll(-70)
        elif count_defects == 4 and CLICK is None:
            CLICK = time.time()
            pyautogui.click()
            CLICK_MESSAGE = "LEFT CLICK" 
    else:
        MOVEMENT_START = None

    if CLICK is not None:
        cv2.putText(img, CLICK_MESSAGE, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
        if CLICK < time.time():     #if some time has elapsed from the click,and no new input is received by the camera,then set CLICK to None. This has to be done to receive the next input from the user and take the necessary action.
             CLICK = None

    cv2.putText(img, "Defects: " + str(count_defects), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    cv2.imshow('Contours', drawing)
    cv2.imshow('Gesture', img)

    k = cv2.waitKey(10) #waits for 10ms for a keypress. if the "esc" key (27) is pressed, then k gets the value 27. so,the code execution stops.
    if k == 27:
        break