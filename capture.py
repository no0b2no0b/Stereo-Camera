from vidgear.gears import VideoGear
import cv2
import os

save_dir = "stereo_images"
os.makedirs(save_dir, exist_ok=True)

camL = VideoGear(source=0).start()
camR = VideoGear(source=1).start()

count = 0

while True:
    left = camL.read()
    right = camR.read()
    
    if left is None or right is None:
        break

    right = cv2.resize(right, (left.shape[1], left.shape[0]))

    cv2.imshow("Left Camera", left)
    cv2.imshow("Right Camera", right)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        cv2.imwrite(f"{save_dir}/left_{count}.png", left)
        cv2.imwrite(f"{save_dir}/right_{count}.png", right)
        print(f"Saved pair {count}")
        count += 1

    if key == ord('q'):
        break

camL.stop()
camR.stop()
cv2.destroyAllWindows()
