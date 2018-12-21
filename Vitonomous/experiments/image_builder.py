import numpy as np
import cv2

image1 = cv2.imread('kernels/kernel_s1.png')
image2 = cv2.imread('kernels/kernel_s2.png')
image3 = cv2.imread('kernels/kernel_s3.png')
shape = (16, 16, 3)

# h, w = 3, 20
# color = np.zeros([h, w, 3], dtype=np.uint8)
# gray = np.zeros([h, w], dtype=np.uint8)
# color[0:1, :, 1] = 255
# gray[0:1, :] = 255

color = image1
gray = color[:, :, 0]
mn, mx = np.min(gray), np.max(gray)
topography = gray-np.min(gray)
contour, counts = np.unique(topography, return_counts=True)
contour_deltas = contour[1:] - contour[:-1]
deltas = np.unique(contour_deltas)

# print(image)
# print(image.reshape(16, 16))
print(gray)
print('mn, mx', mn, mx)
print(topography)
print('contour', contour)
print('contour_deltas', contour_deltas)
print('counts', counts)
print('deltas', deltas)

exit()

cv2.imshow('color', color)
cv2.imshow('gray', gray)
cv2.waitKeyEx(1000 * 5)
cv2.destroyAllWindows()

