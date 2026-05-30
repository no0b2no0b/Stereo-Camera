# sourcery skip: hoist-similar-statement-from-if, hoist-statement-from-if, use-fstring-for-concatenation
import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri
from vidgear.gears import VideoGear

# define and start the stream on first source ( For e.g #0 index device)
streamL = VideoGear(source=1, logging=True).start() 

# define and start the stream on second source ( For e.g #1 index device)
streamR = VideoGear(source=2, logging=True).start() 

frame_rate = 30

#Camera speciifications
B = 9 #Distance between cameras [cm]
f = 24.9 #Camera lense's focal length [mm]
alpha = 62 #Camera field of view in horizontal plane [degree]

#Initial values
count = -1

# infinite loop
while True:
    
    frameL = streamL.read()
    # read frames from stream1

    frameR = streamR.read()
    # read frames from stream2

    # check if any of two frame is None
    if frameL is None or frameR is None:
        #if True break the infinite loop
        break

    count += 1

    #Applying HSV filter
    mask_right = hsv.add_HSV_filter(frameR)
    mask_left = hsv.add_HSV_filter(frameL)

    #Result frames after filters
    resR = cv2.bitwise_and(frameR, frameR, mask = mask_right)
    resL = cv2.bitwise_and(frameL, frameL, mask = mask_left)

    #Applying shape recognition
    circle_right = shape.find_circle(frameR, mask_right)
    circle_left = shape.find_circle(frameL, mask_left)

    if np.all(circle_right) is None or np.all(circle_left) is None:
        cv2.putText(frameR, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frameL, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:

        depth = tri.find_depth(circle_right, circle_left, frameR, frameL, B, f, alpha)

        cv2.putText(frameR, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frameL, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frameR, "Distance: " + str(round(depth, 3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(frameL, "Distance: " + str(round(depth, 3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)

        print("Depth: ", depth)

    cv2.imshow("Output Frame1", frameL)
    cv2.imshow("Output Frame2", frameR)
    cv2.imshow("Mask Left", mask_left)
    cv2.imshow("Mask right", mask_right)
    # Show output window of stream1 and stream 2 seperately

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break

    if key == ord("w"):
        #if 'w' key-pressed save both frameA and frameB at same time
        cv2.imwrite("Image-1.jpg", frameA)
        cv2.imwrite("Image-2.jpg", frameB)
        #break   #uncomment this line to break out after taking images

cv2.destroyAllWindows()
# close output window

# safely close both video streams
streamL.stop()
streamR.stop()