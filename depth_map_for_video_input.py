from vidgear.gears import VideoGear
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

# define and start the stream on first source 
stream1 = VideoGear(source=0, logging=True).start() 

# define and start the stream on second source 
stream2 = VideoGear(source=1, logging=True).start() 

stereo = cv2.StereoBM_create(numDisparities = 0, blockSize = 5)


while True:
    
    frameA = stream1.read()

    frameB = stream2.read()

    if frameA is None or frameB is None:
        break
    
    left_image= cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
    right_image= cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)

    depth = stereo.compute(left_image, right_image)

    
    cv2.imshow("Output Frame1", left_image)
    cv2.imshow("Output Frame2", right_image)

    plt.imshow(depth)
    plt.axis('off')
    plt.pause(0.05)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

    if key == ord("w"):
        cv2.imwrite("Image-1.jpg", left_image)
        cv2.imwrite("Image-2.jpg", right_image)
plt.show()

cv2.destroyAllWindows()

stream1.stop()
stream2.stop()
