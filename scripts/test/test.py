import _init_path_
import yaml
import os

from datetime import datetime
from easydict import EasyDict

from keras.callbacks import ModelCheckpoint, TensorBoard

# TODO: assign model import to overall approach


from utils.models.res_unet import *
from utils.utils import *


def test():
    cfg = EasyDict(yaml.load(open('test.cfg')))

    # configure callbacks

    # TODO: more advanced chechpoint configuration

    ######################## TODO here should be tf of keras model
    # build model
    model = build(cfg.model.parameters)
    # print model summary
    model.summary()
    # define optimizer
    optimizer = configure_optimizer(cfg.optimizer)
    # compile model
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

    # set augmentation
    train_aug = configure_augmentation(cfg.augmentation)
    train_generator = configure_dataset(cfg.dataset, train_aug)


    res = model.evaluate_generator(train_generator)
    print(res)

if __name__ == "__main__":
    test()


