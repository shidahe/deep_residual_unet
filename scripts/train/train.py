import _init_path_


from datetime import datetime
from easydict import EasyDict

from keras.callbacks import ModelCheckpoint, TensorBoard

# TODO: assign model import to overall approach


from utils.models.res_unet import *
from utils.utils import *

cur_path = os.path.abspath(__name__)

def train():
    cfg = EasyDict(yaml.load(open('train.cfg')))
    model_filename = "{model_name}-{dataset_name}".format(model_name=cfg.model.name, dataset_name=cfg.dataset) + \
                     "-{epoch:02d}-{val_acc:.4f}.hdf5"

    # create output folder structure
    # TODO: create more advanced folder structure divided by model name, dataset and so on
    model_output_folder_path = os.path.join(cur_path, "..", "output", datetime.today().strftime("result_%d-%m-%y:%H.%M.%S"))
    os.makedirs(model_output_folder_path)

    # configure callbacks

    # TODO: more advanced checkpoint configuration

    tensorboard = TensorBoard(log_dir=os.path.join(model_output_folder_path, "logs"))
    model_checkpoint = ModelCheckpoint(os.path.join(cur_path, "..", "output", model_filename), monitor='loss',
                                       save_best_only=True, verbose=True)

    # build model
    model = build(cfg.model.parameters)
    # print model summary
    model.summary()
    # define optimizer
    optimizer = configure_optimizer(cfg.optimizer)
    # compile model
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])
    # TODO: improve save current config to output folder
    dump_cfg(os.path.join(model_output_folder_path, "train.cfg"), cfg)
    # set augmentation
    train_aug = configure_augmentation(cfg.augmentation)
    train_generator = configure_dataset(cfg.dataset, train_aug)

    model.fit_generator(train_generator, steps_per_epoch=cfg.steps_per_epoch,
                        epochs=cfg.num_epochs,
                        callbacks=[tensorboard, model_checkpoint]
                        )


if __name__ == "__main__":
    train()


