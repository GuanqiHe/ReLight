import led_calibration as led_c
import numpy as np
import cv2


tar=cv2.imread("data/tar.JPG")
tar_pos=led_c.ColorCardPosExtractClick(tar,save="data/tar")
print(tar_pos.shape)
