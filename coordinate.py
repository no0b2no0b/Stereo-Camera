from vidgear.gears import VideoGear
import cv2
import numpy as np
from ultralytics import YOLO

# parameters
BASELINE_CM = 9.0      
FOCAL_PX = 914.0      


model = YOLO("yolov8n.pt")


camL = VideoGear(source=0).start()
camR = VideoGear(source=1).start()

def get_center(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def compute_3d(cxL, cyL, cxR, w, h, disparity):
    if disparity <= 0:
        return None, None, None

    cx0 = w / 2   
    cy0 = h / 2

    Z = (FOCAL_PX * BASELINE_CM) / disparity  # depth in cm
    X = ((cxL - cx0) * Z) / FOCAL_PX
    Y = ((cyL - cy0) * Z) / FOCAL_PX

    return X, Y, Z

while True:
    left = camL.read()
    right = camR.read()

    if left is None or right is None:
        break

    # resize right to same shape
    right = cv2.resize(right, (left.shape[1], left.shape[0]))
    h, w = left.shape[:2]

    
    preds_left = model(left, verbose=False)[0]
    preds_right = model(right, verbose=False)[0]

    left_boxes = preds_left.boxes
    right_boxes = preds_right.boxes

    # Process LEFT detections
    for boxL in left_boxes:
        x1L, y1L, x2L, y2L = map(int, boxL.xyxy[0])
        clsL = int(boxL.cls[0])

        cxL, cyL = get_center((x1L, y1L, x2L, y2L))

        # Find matching box in right image
        match_box = None
        min_vertical_diff = 9999

        for boxR in right_boxes:
            clsR = int(boxR.cls[0])
            if clsR != clsL:
                continue

            x1R, y1R, x2R, y2R = map(int, boxR.xyxy[0])
            cxR, cyR = get_center((x1R, y1R, x2R, y2R))

            if abs(cyL - cyR) < min_vertical_diff:
                min_vertical_diff = abs(cyL - cyR)
                match_box = (x1R, y1R, x2R, y2R, cxR, cyR)

        if match_box is None:
            continue

        x1R, y1R, x2R, y2R, cxR, cyR = match_box

        # Compute disparity
        disparity = cxL - cxR

        # COMPUTE 3D COORDINATES 
        X, Y, Z = compute_3d(cxL, cyL, cxR, w, h, disparity)

        # Draw on LEFT frame
        cv2.rectangle(left, (x1L, y1L), (x2L, y2L), (0, 255, 0), 2)

        if Z is not None:
            text = f"{model.names[clsL]}: {Z:.2f}cm | X:{X:.1f} Y:{Y:.1f}"
            cv2.putText(left, text, (x1L, y1L - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.circle(left, (cxL, cyL), 5, (0, 0, 255), -1)

        # Draw matching box on RIGHT frame
        cv2.rectangle(right, (x1R, y1R), (x2R, y2R), (255, 0, 0), 2)
        cv2.circle(right, (cxR, cyR), 5, (0, 0, 255), -1)

    
    cv2.imshow("Left Camera", left)
    cv2.imshow("Right Camera", right)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camL.stop()
camR.stop()
cv2.destroyAllWindows()
