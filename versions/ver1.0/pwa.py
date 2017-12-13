#!/usr/bin/env python
""" Compute PWA weights and extract PWA descriptors.

See the DESCRIPTIONS part of the README for more details.
"""


from __future__ import division, print_function


__all__ = ['PWA']
__author__ = 'Hao Zhang'
__copyright__ = 'Copyright @2017 LAMDA'
__date__ = '2017-09-15'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-09-15'
__version__ = '1.0'


import os
import sys
if sys.version[0] == '2':
    input = raw_input

import numpy as np
import sklearn.preprocessing
import sklearn.decomposition


class PWA(object):
    """Compute PWA weights and extract PWA descriptors.

    Attributes:
        _path (dict of (str, str)): Paths used in the program.
        _n_detectors [150] (int): Number of part detectors.
        _dimension [4096] (int): Dimension of the final descriptor.
        _size (int): Size of the dataset.
        _image_names (list of str): Oxford image names.
        _all_pool5 (list of np.ndarray): Pool5 feature for each images, each
            item is a HW*D matrix.
        _selected_channels (np.ndarray): Channel indices selected as part
            detectors.
        _all_weighted_pooled_descriptor (list of np.ndarray): Weighted-pooled
            descriptor for each image.
    """
    def __init__(self, paths, n_detectors=150, dimension=4096):
        self._paths = paths
        self._n_detectors = n_detectors
        self._dimension = dimension

        self._size = None
        self._image_names = []
        self._all_pool5 = []
        self._selected_channels = None
        self._all_weighted_pooled_descriptor = []

    def loadPool5(self):
        """Load pool5 features from disk.

        Names and features for all image are stored in self._image_names and
        self._all_pool5. Size the of the dataset is stored in self._size.
        """
        print('Load raw pool5 features.')
        # Only keep names, not .csv extension.
        self._image_names = [
            f[:-4] for f in os.listdir(self._paths['pool5'])
            if os.path.isfile(os.path.join(self._paths['pool5'], f))]
        self._size = len(self._image_names)

        for i, image_name in enumerate(self._image_names):
            if i % 100 == 0:
                print('Process %d/%d' % (i, self._size))
            xi = np.genfromtxt(os.path.join(
                self._paths['pool5'], image_name + '.csv'), delimiter=',')
            if xi.ndim == 1:  # xi is a row vector
                xi = xi[np.newaxis, :]
            assert xi.shape[1] == 512
            self._all_pool5.append(xi)
        assert len(self._all_pool5) == self._size

    def selectChannels(self):
        """Compute variance and select part detectors.

        First we compute pooled descriptor for each image, and then compute
        vairance among them, and select the top self._n_detectors.
        The indices (starting at 0) of the selected detectors are stored
        in self._selected_channels.
        """
        print('Compute pooled descriptor.')
        all_pooled_descriptor = []
        for i, xi in enumerate(self._all_pool5):
            if i % 100 == 0:
                print('Process %d/%d' % (i, self._size))
            pooled_descriptor = np.sum(xi, axis=0, keepdims=True)
            assert pooled_descriptor.shape == (1, 512)
            all_pooled_descriptor.append(pooled_descriptor)
        assert len(all_pooled_descriptor) == self._size

        print('Select part detector')
        channel_var = np.var(np.vstack(all_pooled_descriptor), axis=0)
        assert channel_var.shape == (512,)
        self._selected_channels = np.argsort(
            channel_var)[::-1][:self._n_detectors]
        np.save(os.path.join(self._paths['channels'], 'indices'),
                self._selected_channels)

    def pooledDescriptor(self):
        """Compute weighted-pooled descriptor for each image.

        The selected spatial map is l2-normalized and sqrt-normalized to form
        weight. Result is stored in self._pooled_descriptor.
        """
        print('Compute pooled descriptors.')
        for i, xi in enumerate(self._all_pool5):
            if i % 100 == 0:
                print('Process %d/%d' % (i, self._size))
            weighted_pooled_descriptor = []
            for j in xrange(self._n_detectors):
                weight_j = xi[:, self._selected_channels[j]]
                assert weight_j.ndim == 1
                weight_j = sklearn.preprocessing.normalize(
                    weight_j[np.newaxis, :])
                weight_j = np.sqrt(weight_j)
                assert weight_j.ndim == 2 and weight_j.shape[0] == 1
                weighted_pooled_descriptor.append(np.sum(
                    weight_j.T * xi, axis=0, keepdims=True))
            stacked = np.hstack(weighted_pooled_descriptor)
            assert stacked.shape == (1, 512 * self._n_detectors)
            self._all_weighted_pooled_descriptor.append(stacked)

    def finalDescriptor(self):
        """Compute the final descriptor for each image for retrieval.

        l2-normalization and PCA whitening are used to form the final
        descriptor.
        """
        print('l2-normalization and PCA whitening.')
        final_descriptor = sklearn.preprocessing.normalize(
            np.vstack(self._all_weighted_pooled_descriptor))
        assert final_descriptor.shape[1] == 512 * self._n_detectors
        pca_model = sklearn.decomposition.PCA(n_components=self._dimension,
                                              whiten=True)
        final_descriptor = pca_model.fit_transform(final_descriptor)

        print('Save final descriptors into disk.')
        for i in xrange(final_descriptor.shape[0]):
            if i % 100 == 0:
                print('Process %d/%d' % (i, self._dimension))
            descriptor_i = final_descriptor[i]
            assert descriptor_i.shape == (self._dimension,)
            np.save(self._paths['descriptor'] + self._image_names[i],
                    descriptor_i)


def main():
    """Main function for the program."""
    project_root = '/data/zhangh/project/pwa/'
    paths = {
        'descriptor':  os.path.join(project_root, 'data/descriptor/all/'),
        'pool5': os.path.join(project_root, 'data/pool5/data/'),
        'channels':  os.path.join(project_root, 'data/channels/'),
    }
    pwa = PWA(paths)
    pwa.loadPool5()
    pwa.selectChannels()
    pwa.pooledDescriptor()
    pwa.finalDescriptor()


if __name__ == '__main__':
    main()
