import cv2
import mediapipe as mp
import math
from FacialDetection import EmotionRecognition


if __name__ == "__main__":
    face_detection = EmotionRecognition()
    face_detection.run()