import sys
import dlib
import numpy as np
from skimage import io
from align import Align
import cv2


# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"

# Take the image file name from the command line
file_name = sys.argv[1]

# Load the image
image = io.imread(file_name)
align = Align(predictor_model)
img=align.align(96,image)

