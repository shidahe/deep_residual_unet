import importlib
import yaml

from keras.optimizers import *
from keras.preprocessing.image import *

smooth = 1.


def configure_augmentation(cfg):
    """
    return
    :param cfg: augmentation fields of configuration
    :return: ImageDataGenerator
    """
    return ImageDataGenerator(featurewise_center=cfg.featurewise_center,
                              samplewise_center=cfg.samplewise_center,
                              featurewise_std_normalization=cfg.featurewise_std_normalization,
                              samplewise_std_normalization=cfg.samplewise_std_normalization,
                              zca_whitening=cfg.zca_whitening,
                              zca_epsilon=cfg.zca_epsilon,
                              rotation_range=cfg.rotation_range,
                              width_shift_range=cfg.width_shift_range,
                              height_shift_range=cfg.height_shift_range,
                              shear_range=cfg.shear_range,
                              zoom_range=cfg.zoom_range,
                              channel_shift_range=cfg.channel_shift_range,
                              fill_mode=cfg.fill_mode,
                              cval=cfg.cval,
                              horizontal_flip=cfg.horizontal_flip,
                              vertical_flip=cfg.vertical_flip,
                              rescale=cfg.rescale)


def configure_optimizer(cfg):
    keras_optimizers = [item.__name__ for item in Optimizer.__subclasses__()]
    if cfg.name in keras_optimizers:
        try:
            return eval("{}(**{})".format(cfg.name, cfg.parameters))
        except TypeError as ex:
            print("Error during configuring optimizer: {}".format(ex))
            exit(-1)
    else:
        print("No such optimizer, please refer to keras docs: http://keras.io/")
        exit(-1)


def configure_dataset(cfg, augm):
    keras_iterators = [item.__name__ for item in Iterator.__subclasses__()]

    if cfg.iterator.name in keras_iterators:
        try:
            return eval("{}(image_data_generator=augm, **{})".format(cfg.name, cfg.parameters))
        except TypeError as ex:
            print("Error during configuring optimizer: {}".format(ex))
            raise
    else:
        try:
            iterator = importlib.import_module("utils.iterators.{}".format(cfg.iterator.name))
            return eval("iterator.{}(image_data_generator=augm, **{})".format(cfg.iterator.name, cfg.iterator.parameters))
        except ImportError as ex:
            print("Unknown iterator module: {}".format(ex))
            raise


def dump_cfg(filepath, cfg):
    with open(filepath, "w+") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


