import json

from keras.optimizers import *
from keras.preprocessing.image import *
import numpy as np
import cv2

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS

cur_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(cur_path, '..', 'models', 'resnet50_pascal_05.pb')

def prepare_points_xy(points_raw):
    points = []
    for j in points_raw:
        if j != 'Skip':
            res = {}
            for key in j:
                polygons = []
                try:
                    type(j[key][0])
                except:
                    print(j)

                if type(j[key][0]) == list:
                    for pp in j[key]:
                        polygons.append({'x': np.array([h['x'] for h in pp]), 'y': np.array([h['y'] for h in pp])})
                else:
                    polygons.append({'x': np.array([h['x'] for h in j[key]]), 'y': np.array([h['y'] for h in j[key]])})
                res[key] = polygons
            points.append(res)
    return points


class LabelBoxXYIterator(Iterator):

    def __init__(self, load_dir=None, image_data_generator=None,
                 image_shape=(300, 300),
                 color_mode='colorful',
                 interpolation='nearest',
                 class_mode='binary',
                 shuffle=True,
                 seed=None,
                 batch_size=32,
                 mode='train'
                 ):
        for i in os.listdir(load_dir):
            if i.endswith('.json'):
                self.file_name = i
        if not self.file_name:
            raise Exception('json file not found')

        with open(load_dir + '/' + self.file_name) as f:
            self.data = json.load(f)

        if mode == 'train':
            mode_file_name = 'train.txt'
        else:
            mode_file_name = 'test.txt'
        with open(os.path.join(load_dir, 'ImageSets', mode_file_name), 'r') as f:
            mode_list = [i.replace('\n', '') for i in f.readlines()]

        path = load_dir + '/JPEGImages/'
        self.filenames = [path + i['External ID'] for i in self.data if i['External ID'] in mode_list]
        self.points = prepare_points_xy([i['Label'] for i in self.data if i['External ID'] in mode_list])

        self.image_shape = tuple(image_shape)
        self.color_mode = color_mode
        self.interpolation = interpolation
        self.data_format = K.image_data_format()

        if image_data_generator:
            self.image_data_generator = image_data_generator
        else:
            self.image_data_generator = None
        self.save_to_dir = None
        self.save_prefix = None
        self.save_format = None
        self.class_mode = class_mode

        self.samples = len(self.filenames)
        shuffle = shuffle
        super(LabelBoxXYIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def load_img(self, path, grayscale=False, target_size=None,
                 interpolation='nearest'):
        """Loads an image into PIL format.

        # Arguments
            path: Path to image file
            grayscale: Boolean, whether to load the image as grayscale.
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.

        # Returns
            A PIL Image instance.

        # Raises
            ImportError: if PIL is not available.
            ValueError: if interpolation method is not supported.
        """
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        img = pil_image.open(path)
        if grayscale:
            if img.mode != 'L':
                img = img.convert('L')
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
        original_size = img.size
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        return img, original_size


    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape + (3,), dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        original_shapes = []
        for i, j in enumerate(index_array):
            fpath = self.filenames[j]
            img, original_shape = self.load_img(fpath,
                           grayscale=grayscale,
                           target_size=self.image_shape,
                           interpolation=self.interpolation)
            original_shapes.append(original_shape)
            x = img_to_array(img, data_format=self.data_format)
            # if self.image_data_generator:
            #     x = self.image_data_generator.random_transform(x)
            #     x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(int(1e7)),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels

        batch_y = self.build_mask(index_array, original_shapes).astype(K.floatx())

        return batch_x, batch_y

    def build_mask(self, index_array, original_shapes):
        masks = [np.zeros(shape[::-1], dtype='float32') for shape in original_shapes]
        for k, ind in enumerate(index_array):
            # TODO: hardcoded class name CORE
            for polygon in self.points[ind]['Core']:
                x = polygon['x']
                y = polygon['y']
                # normalize
                cv2.fillPoly(masks[k], [np.array(list(zip(x, y)))], 1)
        return np.array([cv2.resize(mask[::-1], self.image_shape) for mask in masks])[:,:,:, None]