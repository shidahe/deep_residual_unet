import json

import cv2
from keras.preprocessing.image import *

cur_path = os.path.abspath(os.path.dirname(__file__))

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


def prepare_points_xy(points_raw):
    points = []
    for num, j in enumerate(points_raw):
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

    def __init__(self, dataset_name=None, image_data_generator=None,
                 imageset_file='train',
                 image_shape=(300, 300),
                 color_mode='colorful',
                 interpolation='nearest',
                 shuffle=True,
                 seed=None,
                 batch_size=32,
                 skip_without_mask=True
                 ):
        if dataset_name is None:
            raise FileNotFoundError("Specify dataset name")

        self.dataset_folder = os.path.join(cur_path, "..", "..", "dataset", dataset_name)

        self.skip_without_mask = skip_without_mask
        for i in os.listdir(self.dataset_folder):
            if i.endswith('.json'):
                self.file_name = i
        if not self.file_name:
            raise Exception('json file not found')

        with open(os.path.join(self.dataset_folder, self.file_name)) as f:
            self.data = json.load(f)

        self.imageset_file = imageset_file + '.txt'

        with open(os.path.join(self.dataset_folder, 'ImageSets', self.imageset_file), 'r') as f:
            mode_list = [i.replace('\n', '') for i in f.readlines()]

        path = os.path.join(self.dataset_folder, "JPEGImages")


        self.points = prepare_points_xy([i['Label']
                                         for i in self.data if i['External ID'] in mode_list and i['Label'] != 'Skip'])
        self.filenames = [os.path.join(path, i['External ID'])
                          for i in self.data if i['External ID'] in mode_list and i['Label'] != 'Skip']

        self.image_shape = tuple(image_shape)
        self.color_mode = color_mode
        self.interpolation = interpolation
        self.data_format = K.image_data_format()

        if image_data_generator:
            self.image_data_generator = image_data_generator
        else:
            self.image_data_generator = None

        self.samples = len(self.filenames)
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
                if self.skip_without_mask is None:
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
        batch_x = []
        grayscale = self.color_mode == 'grayscale'
        original_shapes = []

        for i, j in enumerate(index_array):
            fpath = self.filenames[j]
            img, original_shape = self.load_img(fpath,
                                                grayscale=grayscale,
                                                target_size=self.image_shape,
                                                interpolation=self.interpolation)
            original_shapes.append(original_shape)
            x = img_to_array(img, data_format=self.data_format)
            batch_x.append(x)

        batch_y = self.build_mask(index_array, original_shapes)
        for k in range(len(batch_y)):
            if self.image_data_generator:
                txy = self.image_data_generator.random_transform(np.append(batch_x[k], batch_y[k], axis=2))
                batch_y[k] = txy[:, :, 3, None]
                batch_x[k] = self.image_data_generator.standardize(txy[:, :, :3])

        if self.skip_without_mask is not None:
            x, y = self.crop(batch_x, batch_y)
            return np.array(x), np.array(y)
        else:
            return np.array(batch_x), np.array(batch_y)

    def build_mask(self, index_array, original_shapes):
        masks = [np.zeros(shape[::-1], dtype='float32') for shape in original_shapes]
        for k, ind in enumerate(index_array):
            # TODO: hardcoded class name CORE
            for polygon in self.points[ind]['Core']:
                x = polygon['x']
                y = polygon['y']
                cv2.fillPoly(masks[k], [np.array([x, y]).T], 1)
            masks[k] = masks[k][::-1, :]
            if self.skip_without_mask is None:
                masks[k] = cv2.resize(masks[k], self.image_shape)

        for k, ind in enumerate(index_array):
            masks[k] = np.expand_dims(masks[k], 2).astype(K.floatx())
        return masks

    def crop(self, x, y):
        hx, hy = self.image_shape

        new_x, new_y = [], []
        for k, mask in enumerate(y):
            flag = True

            if x[k].shape[0] < self.image_shape[0] or x[k].shape[1] < self.image_shape[1]:
                cropped_img = cv2.resize(x[k], self.image_shape)
                cropped_mask = np.expand_dims(cv2.resize(mask, self.image_shape), 2)
                flag = False
            while flag:
                x_ind = np.random.randint(0, x[k].shape[0] - hx)
                y_ind = np.random.randint(0, x[k].shape[1] - hy)
                cropped_mask = mask[x_ind: x_ind + hx, y_ind: y_ind + hy]

                if self.skip_without_mask and cropped_mask.mean() > 5.0 / (hx * hy):
                    cropped_img = x[k][x_ind: x_ind + hx, y_ind: y_ind + hy]
                    flag = False
                elif not self.skip_without_mask:
                    cropped_img = x[k][x_ind: x_ind + hx, y_ind: y_ind + hy]
                    flag = False
            new_x.append(cropped_img)
            new_y.append(cropped_mask)

        return np.array(new_x), np.array(new_y)
