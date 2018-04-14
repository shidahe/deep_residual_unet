import os
import importlib

import yaml
import keras
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

keras_optimizers = [item.__name__ for item in Optimizer.__subclasses__()]


def configure_augmentation(cfg):
    """
    return
    :param cfg: augmentation fields of configuration
    :return: ImageDataGenerator
    """
    return ImageDataGenerator(**cfg)


def configure_optimizer(cfg):

    if cfg.name in keras_optimizers:
        try:
            return eval("{}(**{})".format(cfg.name, cfg.parameters))
        except TypeError as ex:
            print("Error during configuring optimizer: {}".format(ex))
            raise ex
    else:
        print("No such optimizer, please refer to keras docs: http://keras.io/")
        exit(-1)


def configure_generator(dataset_name, cfg, augm):
    try:
        iterator = importlib.import_module("utils.iterators.{}".format(cfg.iterator))
        return eval(
            "iterator.{}(dataset_name=dataset_name, image_data_generator=augm, **{})".
                format(cfg.iterator, cfg.parameters))
    except ImportError as ex:
        print("Unknown iterator module: {}".format(ex))
        raise


def configure_model(model_cfg):
    try:
        model_lib = importlib.import_module("utils.models.{}".format(model_cfg.name))
        return eval("model_lib.build({})".format(model_cfg.parameters))
    except ImportError as ex:
        print("Unknown model: {}".format(ex))
        raise


def dump_cfg(filepath, cfg):
    with open(filepath, "w+") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    diff = 0.5 * K.sum(y_pred_f - y_true_f)
    return (2. * intersection - diff + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# TODO: should it be deprecated?
def save_keras_model(model, name):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(cur_path, 'weights', name)
    model.save(model_path)


def load_keras_model(name):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(cur_path, '..', 'demo', name)
    return keras.models.load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})


