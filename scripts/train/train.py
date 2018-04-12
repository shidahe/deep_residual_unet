import _init_path_

from datetime import datetime
from easydict import EasyDict

from keras.callbacks import ModelCheckpoint, TensorBoard

from utils.utils import *

cur_path = os.path.abspath(os.path.dirname(__file__))


def train():
    cfg = EasyDict(yaml.load(open('train.cfg')))
    model_filename = "{model_name}-{dataset_name}".format(model_name=cfg.model.name,
                                                          dataset_name=cfg.dataset.folder_name) + \
                     "-{epoch:02d}-({val_loss:.4f}).h5py"

    # create output folder structure
    # TODO: create more advanced folder structure divided by model name, dataset and so on
    model_output_folder_path = os.path.join(cur_path, "..", "output", cfg.model.name, cfg.dataset.folder_name,
                                            datetime.today().strftime("%d-%m-%y_%H.%M.%S"))
    os.makedirs(model_output_folder_path)

    # configure callbacks

    # TODO: more advanced checkpoint configuration

    tensorboard = TensorBoard(log_dir=os.path.join(model_output_folder_path, "tensorboard_logs"))
    model_checkpoint = ModelCheckpoint(os.path.join(model_output_folder_path, model_filename), monitor='val_loss',
                                       save_best_only=True, verbose=True)

    # build model
    model = configure_model(cfg.model)
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
    train_generator = configure_generator(cfg.dataset, train_aug)
    val_generator = configure_generator(cfg.dataset, None)

    model.fit_generator(train_generator, steps_per_epoch=cfg.steps_per_epoch,
                        epochs=cfg.num_epochs, validation_data=val_generator,
                        validation_steps=250,
                        callbacks=[tensorboard, model_checkpoint]
                        )


if __name__ == "__main__":
    train()


