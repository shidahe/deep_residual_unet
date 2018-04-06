import _init_path_
import numpy as np
import cv2
import keras
from utils.utils import *

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc_demo = cv2.VideoWriter_fourcc(*'XVID')

cam = cv2.VideoCapture('brevno_demo.avi')
_, img = cam.read()
out_shape = img.shape[:2]
cam.release()
out = cv2.VideoWriter('brevno_demo_mask.avi', fourcc, 20.0, out_shape[::-1])
out_demo = cv2.VideoWriter('brevno_demo_output.avi', fourcc_demo, 20.0, out_shape[::-1])
print(out_shape[::-1])
cam = cv2.VideoCapture('brevno_demo.avi')

model = load_keras_model('keras.h5')

while True:
    ret, img = cam.read()
    shape = img.shape[:2]
    if ret:
        print(ret)
        img_r = cv2.resize(img, (512, 512))
        print(shape)

        mask_orig = cv2.resize(model.predict(np.array([img_r]))[0], tuple(shape[::-1]))
        # print(mask_orig.shape)

        cv2.imshow('1', cv2.resize(mask_orig, None, fx=0.5, fy=0.5))
        mask_orig = np.array([mask_orig * 255] * 3).transpose((1, 2, 0)).astype('uint8')
        out.write(mask_orig)

        # out_demo.write(np.append(img, (255 * mask_orig * 0.5)[:, :, None], axis=2).astype('uint8'))
        out_demo.write(cv2.addWeighted(img, 1, mask_orig, 0.5, 0))
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    else:
        break


cv2.destroyAllWindows()
out.release()
out_demo.release()
