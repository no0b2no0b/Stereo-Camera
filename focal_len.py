import math

W = 30  # image width
FOV = 62  # degrees (typical webcam)

f = W / (2 * math.tan(math.radians(FOV/2)))
print("Focal length =", f)
