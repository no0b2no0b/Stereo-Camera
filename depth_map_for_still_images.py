import cv2
from matplotlib import pyplot as plt

left_image= cv2.imread("photo_left.jpg", cv2.IMREAD_GRAYSCALE)
right_image= cv2.imread("photo_right.jpg", cv2.IMREAD_GRAYSCALE)

stereo = cv2.StereoBM_create(numDisparities = 64, blockSize = 5)

depth = stereo.compute(left_image, right_image)


cv2.imshow("Output Frame1", left_image)
cv2.imshow("Output Frame2", right_image)

plt.imshow(depth)
plt.axis('off')
plt.show()
