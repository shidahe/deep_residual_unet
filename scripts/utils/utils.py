import importlib
import yaml

from keras.optimizers import *
from keras.preprocessing.image import *
import keras
smooth = 1.


def configure_augmentation(cfg):
    """
    return
    :param cfg: augmentation fields of configuration
    :return: ImageDataGenerator
    """
    return ImageDataGenerator(**cfg)


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
            iterator = importlib.import_module("utils.iterators.{}".format(cfg.iterator.name))
            return eval("{}(image_data_generator=augm, **{})".format(cfg.iterator.name, cfg.iterator.parameters))
        except TypeError as ex:
            print("Error during configuring optimizer: {}".format(ex))
            raise
    else:
        try:
            iterator = importlib.import_module("utils.iterators.{}".format(cfg.iterator.name))
            return eval(
                "iterator.{}(image_data_generator=augm, **{})".format(cfg.iterator.name, cfg.iterator.parameters))
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

#TODO should it be deprecated?
def save_keras_model(model, name):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(cur_path, 'weights', name)
    model.save(model_path)


def load_keras_model(name):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(cur_path, 'weights', name)
    return keras.models.load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})


