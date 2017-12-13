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
__updated__ = '2017-09-17'
__version__ = '1.3'


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
        _paths, dict of (str, str): Paths used in the program.
        _n_detectors, int [150]: Number of part detectors.
        _dimension, int [4096]: Dimension of the final descriptor.
        _size, dict of (str, int): Size of the full image and cropped query
            images dataset.
        _names, dict of (str, list of str): Full image and cropped query image
            names.
        _pool5, dict of (str, list of np.ndarray): Pool5 feature for full
            image and cropped query images, each item is a HW*D matrix.
        _selected_channels, np.ndarray: Channel indices selected as part
            detectors.
        _weighted_pooled_descriptor, dict of (str, list of np.ndarray):
            Weighted-pooled descriptor for each full image/cropped query image.
    """
    def __init__(self, paths, n_detectors=150, dimension=4096):
        self._paths = paths
        self._n_detectors = n_detectors
        self._dimension = dimension

        self._size = {'all': None, 'crop': None}
        self._names = {'all': [], 'crop': []}
        self._pool5 = {'all': [], 'crop': []}
        self._selected_channels = None
        self._weighted_pooled_descriptor = {'all': [], 'crop': []}

    def loadPool5(self, image_type):
        """Load pool5 features from disk.

        Names and features for all image are stored in self._image_names and
        self._all_pool5. Size the of the dataset is stored in self._size.

        Args:
            image_type, str: 'query' or 'all', indicating to load whether full
                image or cropped query image pool5 features.
        """
        print('Load raw pool5 features for %s.' % image_type)
        # Only keep names, not .csv extension.
        self._names[image_type] = [
            f[:-4] for f in os.listdir(self._paths['pool5'][image_type])
            if os.path.isfile(os.path.join(self._paths['pool5'][image_type],f))]
        self._size[image_type] = len(self._names[image_type])

        for i, image_name in enumerate(self._names[image_type]):
            if i % 100 == 0:
                print('Process %d/%d' % (i, self._size[image_type]))
            xi = np.genfromtxt(os.path.join(self._paths['pool5'][image_type],
                                            image_name + '.csv'), delimiter=',')
            if xi.ndim == 1:  # xi is a row vector
                xi = xi[np.newaxis, :]
            assert xi.shape[1] == 512
            self._pool5[image_type].append(xi)
        assert len(self._pool5[image_type]) == self._size[image_type]

    def selectChannels(self):
        """Compute variance and select part detectors.

        First we compute pooled descriptor for each image, and then compute
        vairance among them, and select the top self._n_detectors.
        The indices (starting at 0) of the selected detectors are stored
        in self._selected_channels.
        """
        print('Compute pooled descriptor.')
        all_pooled_descriptor = []
        for i, xi in enumerate(self._pool5['all']):
            if i % 100 == 0:
                print('Process %d/%d' % (i, self._size['all']))
            pooled_descriptor = np.sum(xi, axis=0, keepdims=True)
            assert pooled_descriptor.shape == (1, 512)
            all_pooled_descriptor.append(pooled_descriptor)
        assert len(all_pooled_descriptor) == self._size['all']

        print('Select part detector')
        channel_var = np.var(np.vstack(all_pooled_descriptor), axis=0)
        assert channel_var.shape == (512,)
        self._selected_channels = np.argsort(
            channel_var)[::-1][:self._n_detectors]
        assert self._selected_channels.shape == (self._n_detectors,)
        np.save(os.path.join(self._paths['channels'], 'indices'),
                self._selected_channels)

    def pooledDescriptor(self, image_type):
        """Compute weighted-pooled descriptor for each image.

        The selected spatial map is l2-normalized and sqrt-normalized to form
        weight. Result is stored in self._pooled_descriptor.

        Args:
            image_type, str: 'query' or 'all', indicating to load whether full
                image or cropped query image pool5 features.
        """
        print('Compute pooled descriptors for %s.' % image_type)
        for i, xi in enumerate(self._pool5[image_type]):
            if i % 100 == 0:
                print('Process %d/%d' % (i, self._size[image_type]))
            weighted_pooled_descriptor = []
            for j in xrange(self._n_detectors):
                weight_j = xi[:, self._selected_channels[j]]
                assert weight_j.ndim == 1
                # weight_j = weight_j / np.sqrt(np.sum(weight_j ** 2))
                # weight_j = np.sqrt(weight_j[np.newaxis, :])
                weight_j = sklearn.preprocessing.normalize(
                    weight_j[np.newaxis, :])
                assert weight_j.ndim == 2 and weight_j.shape[0] == 1
                weighted_pooled_descriptor.append(np.sum(
                    weight_j.T * xi, axis=0, keepdims=True))
            stacked = np.hstack(weighted_pooled_descriptor)
            assert stacked.shape == (1, 512 * self._n_detectors)
            self._weighted_pooled_descriptor[image_type].append(stacked)

    def finalDescriptor(self):
        """Compute the final descriptor for each image for retrieval.

        l2-normalization and PCA whitening are used to form the final
        descriptor.
        """
        print('l2-normalization and PCA whitening.')
        descriptor = {
            'all': sklearn.preprocessing.normalize(
                np.vstack(self._weighted_pooled_descriptor['all'])),
            'crop': sklearn.preprocessing.normalize(
                np.vstack(self._weighted_pooled_descriptor['crop'])),
        }
        assert descriptor['all'].shape[1] == 512 * self._n_detectors
        assert descriptor['crop'].shape[1] == 512 * self._n_detectors
        pca_model = sklearn.decomposition.PCA(n_components=self._dimension,
                                              whiten=True)
        descriptor['all'] = pca_model.fit_transform(descriptor['all'])
        descriptor['crop'] = pca_model.transform(descriptor['crop'])

        for image_type in ['all', 'crop']:
            print('Save final descriptors into disk for %s.' % image_type)
            for i in xrange(descriptor[image_type].shape[0]):
                if i % 100 == 0:
                    print('Process %d/%d' % (i, self._size[image_type]))

                descriptor_i = descriptor[image_type][i]
                assert descriptor_i.shape == (self._dimension,)
                np.save(self._paths['descriptor'][image_type] +
                        self._names[image_type][i], descriptor_i)


def main():
    """Main function for the program."""
    project_root = '/data/zhangh/project/pwa/'
    paths = {
        'channels':  os.path.join(project_root, 'data/channels/'),
        'descriptor': {
            'all': os.path.join(project_root, 'data/descriptor/all/'),
            'crop': os.path.join(project_root, 'data/descriptor/crop/'),
        },
        'pool5': {
            'all': os.path.join(project_root, 'data/pool5/data-all/'),
            'crop': os.path.join(project_root, 'data/pool5/data-crop/'),
        },
    }
    pwa = PWA(paths, n_detectors=10)
    pwa.loadPool5('all')
    pwa.loadPool5('crop')
    pwa.selectChannels()
    pwa.pooledDescriptor('all')
    pwa.pooledDescriptor('crop')
    pwa.finalDescriptor()


if __name__ == '__main__':
    main()
