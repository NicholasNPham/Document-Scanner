from transform import fourPointTransform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import os

"""
Step 1: Edge Detection
"""

# Finding file location.
image = cv2.imread('image/image.png')

# Changing Ratio of Image Original
ratio = image.shape[0] / 500.0
original = image.copy()
image = imutils.resize(image, height = 500)

# Turn Image from Color to GrayScale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Print Step 1
print('Step 1: Edge Detection')

# CV2 DISPLAY
# cv2.imshow('Image', image)
# cv2.imshow("Edge", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
Step 2: Finding Contour of Paper
"""

# Finds the contours in the edged image
# Keeps the largest one, and Initialize the screen contour.
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# Looping over the contours
for c in contours:
    # Estimation of contour
    perimeter = cv2.arcLength(c, True)
    estimate = cv2.approxPolyDP(c, 0.02 * perimeter, True)

    # If estimate contour has four points, screen is found
    if len(estimate) == 4:
        screenContour = estimate
        break

# CV2 Showing Image with Contour
print('Step 2: Finding Contour of Paper')
# cv2.drawContours(image, [screenContour], -1, (0, 255, 0), 2)
# cv2.imshow('Outline', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
Step 3: Apply Perspective Transform & Threshold
"""

# Apply the Four Point Transform to obtain Top-Down View
warped = fourPointTransform(original, screenContour.reshape(4, 2) * ratio)

# Converting the warped image to grayscale then threshold it
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# Show the original and scanned image
print('Step 3: Applying Perspective Transform')
cv2.imshow("Original", imutils.resize(original, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))

cv2.waitKey(0)

"""
Saving Image to File
"""
# Destination File Directory
directory = 'E:\Current Project\Document-Scanner\Scanned'

# Changing Named Saved
filename = 'ScannedImage.png'

# Writing/Downloading the file into the directory with the changed filename.
cv2.imwrite(os.path.join(directory, filename), warped)
print('Imaged Saved')