from PIL import Image
import numpy as np
import cv2

class Segment:
    def __init__(self) -> None:
        pass

    def __getContours(self, thresh):
        """
        Get contours from threshold image

        Args:
            thresh: numpy array

        Returns:
            contours: numpy array
        """
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def getPoints(self, mask):
        """
        Given a mask image, get the approximate points of the mask

        Args:
            masks: numpy array

        Returns:
            points: numpy array
        """
        contours = self.__getContours(mask)
        peri = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.005*peri, True)
        return approx
