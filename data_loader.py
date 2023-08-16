import numpy as np
import struct
from array import array

class MnistDataloader(object):
    def vectorized_result(self, labels):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        vectorized = []
        for label in labels:
            e = np.zeros((10, 1))
            e[label] = 1.0
            vectorized.append(e)
        return vectorized

    def flatten_dataset(self, x, y):
        x = np.reshape(x, (-1, 784))
        y = y[:, :, 0]
        return x, y

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        from pathlib import Path
        print(Path.cwd())
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []

        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            img = img / 255
            images.append(img)

        return self.flatten_dataset(np.array(images), np.array(self.vectorized_result(labels)))

    def load_data(self):
        training_images_filepath = 'dataset/train-images-idx3-ubyte/train-images-idx3-ubyte'
        training_labels_filepath = 'dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
        test_images_filepath = 'dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
        test_labels_filepath = 'dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
        x_train, y_train = self.read_images_labels(training_images_filepath, training_labels_filepath)
        x_test, y_test = self.read_images_labels(test_images_filepath, test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)