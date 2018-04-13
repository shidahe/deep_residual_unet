import _init_path_
import numpy as np
import cv2
import keras
from utils.utils import *
from tqdm import tqdm
from easydict import EasyDict

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

model = load_keras_model('unet-brewno-02--0.9796.h5')

cfg = EasyDict(yaml.load(open('train.cfg')))

def crop_predict(model, img):
    mask = np.zeros(img.shape[:-1])
    mask_counts = np.zeros(img.shape[:-1])
    hx, hy = cfg.dataset.iterator.parameters.image_shape
    h = 128
    shape = img.shape[:-1]
    xs = shape[0]
    ys = shape[1]
    for i in (range(0, xs - hx + h, h)):
        for j in (range(0, ys - hy + h, h)):
            cropped = img[i: i + hx, j: j + hy]
            cropped_shape = cropped.shape
            if cropped_shape[0] != hx or cropped_shape[1] != hy:
                cropped = cv2.resize(cropped, (hx, hy))

            new_img = cropped.reshape(1, cropped.shape[0], cropped.shape[1], 3)

            mask_crop = model.predict(new_img)[0][:, :, 0]

            if cropped_shape[0] != hx or cropped_shape[1] != hy:
                mask_crop = cv2.resize(mask_crop, (cropped_shape[1], cropped_shape[0]))

            mask[i: i + hx, j: j + hy] += mask_crop
            mask_counts[i: i + hx, j: j + hy] += 1

    return mask / mask_counts


crop = False
while True:
    ret, img = cam.read()
    shape = img.shape[:2]
    if ret:
        if crop:
            img_r = img
            img_r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask_orig = crop_predict(model, img_r)

            mask_orig = np.array([mask_orig * 255] * 3).transpose((1, 2, 0)).astype('uint8')
            out.write(mask_orig)

            out_demo.write(cv2.addWeighted(img, 1, mask_orig, 0.5, 0))
        else:
            img_r = cv2.resize(img, (256, 256))
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            mask_orig = cv2.resize(model.predict(np.array([img_r]))[0], tuple(shape[::-1]))
            mask_orig = np.array([mask_orig * 255] * 3).transpose((1, 2, 0)).astype('uint8')
            out.write(mask_orig)

            out_demo.write(cv2.addWeighted(img, 1, mask_orig, 0.5, 0))
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    else:
        break

cv2.destroyAllWindows()
out.release()
out_demo.release()
