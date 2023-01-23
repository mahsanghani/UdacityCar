import cv2.cv2 as cv2

def warper(img, src, dst):
    size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, size, flags=cv2.INTER_NEAREST)
    return warped
