import _init_path_

from datetime import datetime
from easydict import EasyDict

from keras.callbacks import ModelCheckpoint, TensorBoard

from utils.utils import *

cur_path = os.path.abspath(os.path.dirname(__file__))
result_folder = os.path.join(cur_path, "..", "output")
dataset_folder = os.path.join(cur_path, "..", "dataset")


def train():
    cfg = EasyDict(yaml.load(open('train.cfg')))
    model_filename = "{model_name}-{dataset_name}".format(model_name=cfg.model.name,
                                                          dataset_name=cfg.dataset.name) + \
                     "-{epoch:02d}-({val_loss:.4f}).h5py"

    model_output_folder_path = os.path.join(result_folder, cfg.model.name, cfg.dataset.name,
                                            datetime.today().strftime("%d-%m-%y_%H.%M.%S"))
    os.makedirs(model_output_folder_path)

    # configure callbacks

    # TODO: more advanced checkpoint configuration

    tensorboard = TensorBoard(log_dir=os.path.join(model_output_folder_path, "tensorboard_logs"))
    model_checkpoint = ModelCheckpoint(os.path.join(model_output_folder_path, model_filename), monitor='val_loss',
                                       save_best_only=True, verbose=True)

    # build model
    model = configure_model(cfg.model)
    # set augmentation
    train_aug = configure_augmentation(cfg.dataset.augmentation)
    # generators
    train_generator = configure_generator(cfg.dataset.name, cfg.dataset.train, train_aug)
    val_generator = configure_generator(cfg.dataset.name, cfg.dataset.validation, None)
    # define optimizer
    optimizer = configure_optimizer(cfg.optimizer)
    # print model summary
    model.summary()
    # compile model
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])
    # TODO: improve save current config to output folder
    dump_cfg(os.path.join(model_output_folder_path, "train.cfg"), cfg)

    model.fit_generator(train_generator, steps_per_epoch=cfg.dataset.train.steps_per_epoch,
                        epochs=cfg.num_epochs, validation_data=val_generator,
                        validation_steps=cfg.dataset.validation.steps_per_epoch,
                        callbacks=[tensorboard, model_checkpoint]
                        )


if __name__ == "__main__":
    train()


