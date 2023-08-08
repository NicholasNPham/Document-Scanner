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

def fourPointTransform(image, points):

    # Initialize the Four Main Points in Order.
    rect = cornerPoints(points)
    # Separate the order points into diff variables.
    (tl, tr, br, bl) = rect

    # Width is computed with points of bottom and top. Equaling the X coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - bl[1]) ** 2))
    
    # Finds the Max Width between the top points of bottom points
    maxWidth = max(int(widthA), int(widthB))

    # Height is computer with the points of left and right. Equaling to the Y coordinates.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # Finds the max Height between the left and right points of square.
    maxHeight = max(int(heightA), int(heightB))

    # Goes TL, TR, BR, BL
    distance = np.array([
        [0, 0], 
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32')
    
    # Initalize the Matrix then apply warped Perspective.
    matrix = cv2.getPerspectiveTransform(rect, distance)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    # return warped from function.
    return warped
