import numpy as np
import cv2

def cornerPoints(points):

    # TL -> TR -> BR -> BL
    rect = np.zeros((4, 2), dtype='float32')

    sum = points.sum(axis=1)
    
    # Top Left = Smallest Sum
    rect[0] = points[np.argmin(sum)]
    # Bottom Right = Largest Sum
    rect[2] = points[np.argmax(sum)]

    diff = np.diff(points, axis=1)

    # Top Right = Smallest Diff
    rect[1] = points[np.argmin(diff)]
    # Bottom Left = Largest Diff
    rect[2] = points[np.argmax(diff)]

    # Return Ordered Coordinates
    return rect


