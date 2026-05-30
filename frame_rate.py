import cv2
import time

cap = cv2.VideoCapture(0)

prev = time.time()
frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames += 1
    now = time.time()

    # Every 1 second print FPS
    if now - prev >= 1:
        print("FPS:", frames)
        frames = 0
        prev = now

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
