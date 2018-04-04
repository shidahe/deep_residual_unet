"""
https://arxiv.org/pdf/1711.10684.pdf
Road extraction by Deep Residual U-Net
"""

from datetime import datetime
from easydict import EasyDict

from keras.callbacks import ModelCheckpoint, TensorBoard

from scripts.utils.res_unet import *
from scripts.utils.utils import *

cfg = EasyDict(yaml.load(open('train.cfg')))

# hyper parameters

model_filename = "{model_name}-{dataset_name}".format(model_name=cfg.model_name, dataset_name=cfg.dataset) + \
                 "-{epoch:02d}-{val_acc:.4f}.hdf5"

# create output folder structure
# TODO: create more advanced folder structure divided by model name, dataset and so on

model_output_folder_path = os.path.join("..", "models", datetime.today().strftime("result_%d-%m-%y:%H.%M.%S"))
os.makedirs(model_output_folder_path)

# configure callbacks

tensorboard = TensorBoard(log_dir=os.path.join(model_output_folder_path, "logs"))
model_checkpoint = ModelCheckpoint(os.path.join("..", "models", model_filename), monitor='loss',
                                   save_best_only=True, verbose=True)

# build model

model = build_res_unet(input_shape=cfg.input_size)

# print model summary
model.summary()

# define optimizer

optimizer = configure_optimizer(cfg.optimizer)

# compile model

model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

# TODO: save current config to output folder

dump_cfg(os.path.join(model_output_folder_path, "train.cfg"), cfg)

# set augmentation

train_aug = configure_augmentation(cfg.augmentation)

train_gen = PASCALVOCIterator(directory=dataset_folder, target_file="train.txt",
                              image_data_generator=train_aug, target_size=(input_shape[0], input_shape[1]),
                              batch_size=batch_size, classes=classes)

# train model

model.fit_generator(train_gen, steps_per_epoch=300,
                    epochs=50,
                    callbacks=[tensorboard, model_checkpoint]
                    )
