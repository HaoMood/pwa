#!/usr/bin/env python
"""Find features for query images and evaluate the result.

We use the number PC1 projections >0 as the part detector to perform weighted
sum-pooling of pool5 features to obtain the final descriptor.

This program is modified from CroW:
    https://github.com/yahoo/crow/

The compute_ap executable file is modified from VGG:
    http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
To compute the average precision for a ranked list of query christ_church_1,
run:
    ./compute_ap christ_church_1 ranked_list.txt
"""


from __future__ import division, print_function


__all__ = ['EvaluateManager']
__author__ = 'Hao Zhang'
__copyright__ = 'Copyright @2017 LAMDA'
__date__ = '2017-09-26'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-10-02'
__version__ = '2.6'


import argparse
import itertools
import os
import sys
if sys.version[0] == '2':
    filter = itertools.ifilter
    input = raw_input
    map = itertools.imap
    range = xrange
    zip = itertools.izip
import tempfile

import numpy as np
import sklearn.decomposition
import sklearn.neighbors
import sklearn.preprocessing


class EvaluateManager(object):
    """Manager class to compute mAP.

    Attributes:
        _paths, dict of (str, str): Feature paths.
        _n_detectors, int: Number of part detectors.
        _pca_model: sklearn.decomposition.PCA model for final PCA whitening.
    """
    def __init__(self, paths, n_detectors):
        self._paths = paths
        self._n_detectors = n_detectors
        self._pca_model = None

    def fitPca(self, dim, whiten):
        """Compute PCA whitening paramters from the via dataset.

        Args:
            via_pool5_path, str: Pool5 feature path of the via dataset.
            dim, int: Dimension of the final retrieval descriptor.
            whiten, bool: Whether to use whitening.
        """
        print('Compute via dataset part detectors.')
        via_selected_channels = self._selectChannels('via_all_pool5')
        print('Load via dataset descriptors.')
        via_all_descriptors, _ = self._loadFeature('via_all_pool5',
                                                   via_selected_channels)
        via_all_descriptors = np.vstack(via_all_descriptors)
        assert via_all_descriptors.shape[1] == 512 * self._n_detectors

        print('Fit PCA whitening paramters from via dataset.')
        sklearn.preprocessing.normalize(via_all_descriptors, copy=False)
        self._pca_model = sklearn.decomposition.PCA(n_components=dim,
                                                    whiten=whiten)
        self._pca_model.fit(via_all_descriptors)

    def evaluate(self, crop):
        """Evaluate the retrieval results.

        Args:
            crop, bool: Whether to use whitening.
        """
        print('Compute test dataset part detectors.')
        test_selected_channels = self._selectChannels('test_all_pool5')
        print('Load test dataset descriptors.')
        test_all_descriptors, test_all_names = self._loadFeature(
            'test_all_pool5', test_selected_channels)
        test_query_pool5 = 'test_crop_pool5' if crop else 'test_full_pool5'
        test_crop_descriptors, test_crop_names = self._loadFeature(
            test_query_pool5, test_selected_channels)

        test_all_descriptors, test_crop_descriptors = self._normalization(
            test_all_descriptors, test_crop_descriptors)
        assert test_all_descriptors.shape[1] == test_crop_descriptors.shape[1]

        # Iterate queries, process them, rank results, and evaluate mAP.
        all_ap = []
        for i in xrange(len(test_crop_names)):
            knn_model = sklearn.neighbors.NearestNeighbors(
                n_neighbors=len(test_all_descriptors))
            knn_model.fit(test_all_descriptors)
            _, ind = knn_model.kneighbors(
                test_crop_descriptors[i].reshape(1, -1))
            ap = self._getAP(ind[0], test_crop_names[i], test_all_names)
            all_ap.append(ap)
        m_ap = np.mean(np.array(all_ap))
        print('mAP is', m_ap)

    def _selectChannels(self, pool5_path):
        """Compute variance for each dimension of the pooled descriptor and
        select part detectors.

        This is a helper function of fitPca() and evaluate().

        Args:
            pool5_path, str: Path of .npy pool5 features.

        Return:
            selected_channels, np.ndarray of size self._n_detectors: Channel
                indices selected as part detectors.
        """
        # Load pool5 features.
        name_list = sorted([
            f for f in os.listdir(self._paths[pool5_path])
            if os.path.isfile(os.path.join(self._paths[pool5_path], f))])
        m = len(name_list)
        descriptor_list = []
        for i, name_i in enumerate(name_list):
            if i % 100 == 0:
                print('Processing %d/%d' % (i, m))
            # Load pool5 feature.
            pool5_i = np.load(os.path.join(self._paths[pool5_path], name_i))
            assert pool5_i.ndim == 3 and pool5_i.shape[0] == 512
            descriptor_list.append(np.sum(pool5_i,
                                          axis=(1, 2))[np.newaxis, :])

        # Compute pooled descriptor.
        channel_var = np.var(np.vstack(descriptor_list), axis=0)
        assert channel_var.shape == (512,)
        selected_channels = np.argsort(channel_var)[::-1][:self._n_detectors]
        assert selected_channels.shape == (self._n_detectors,)
        return selected_channels

    def _loadFeature(self, pool5_path, selected_channels):
        """Load and process pool5 features into weighted-pooled descriptors.

        This is a helper function of fitPca() and evaluate().

        Args:
            pool5_path, str: Path of .npy pool5 features.
            selected_channels, np.ndarray of size self._n_detectors: Channel
                indices selected as part detectors.

        Return:
            descriptor_list, list of np.ndarray of size 1*512: List of
                weighted-pooled descriptors.
            name_list, list of str: List of image names without extensions.
        """
        name_list = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in os.listdir(self._paths[pool5_path])
            if os.path.isfile(os.path.join(self._paths[pool5_path], f))])
        m = len(name_list)
        descriptor_list = []
        for i, name_i in enumerate(name_list):
            if i % 100 == 0:
                print('Processing %d/%d' % (i, m))
            # Load pool5 feature.
            pool5_i = np.load('%s.npy' % os.path.join(self._paths[pool5_path],
                                                      name_i))
            assert pool5_i.ndim == 3 and pool5_i.shape[0] == 512
            H, W = pool5_i.shape[1], pool5_i.shape[2]

            descriptor_i = []
            for j in range(self._n_detectors):
                weight_j = pool5_i[selected_channels[j], :, :]
                weight_j = np.reshape(weight_j, (1, H * W))
                sklearn.preprocessing.normalize(weight_j, copy=False)
                weight_j = np.sqrt(weight_j)
                weight_j = np.reshape(weight_j, (1, H, W))
                descriptor_i.append(np.sum(weight_j * pool5_i,
                                           axis=(1, 2))[np.newaxis, :])
            descriptor_i = np.hstack(descriptor_i)
            assert descriptor_i.shape == (1, 512 * self._n_detectors)
            descriptor_list.append(descriptor_i)
        return descriptor_list, name_list

    def _normalization(self, test_all_descriptors, test_crop_descriptors):
        """l2 normalize and PCA whitening for test all and cropped query
        descriptors.

        This is a helper function of evaluate().

        Args:
            test_all_descriptors, np.ndarray of m*512: Before normalize.
            test_crop_descriptors, np.ndarray of m*512: Before normalize.

        Return
            test_all_descriptors, np.ndarray of m*512: After normalize.
            test_crop_descriptors, np.ndarray of m*512: After normalize.
        """
        test_all_descriptors = np.vstack(test_all_descriptors)
        test_crop_descriptors = np.vstack(test_crop_descriptors)

        sklearn.preprocessing.normalize(test_all_descriptors, copy=False)
        sklearn.preprocessing.normalize(test_crop_descriptors, copy=False)

        test_all_descriptors = self._pca_model.transform(test_all_descriptors)
        test_crop_descriptors = self._pca_model.transform(test_crop_descriptors)

        sklearn.preprocessing.normalize(test_all_descriptors, copy=False)
        sklearn.preprocessing.normalize(test_crop_descriptors, copy=False)
        return test_all_descriptors, test_crop_descriptors

    def _getAP(self, ind, query_name, all_names):
        """Given a query, compute average precision for the results by calling
        to the compute_ap.

        This is a helper function of evaluate().
        """
        # Generate a temporary file.
        f = tempfile.NamedTemporaryFile(delete=False)
        temp_filename = f.name
        f.writelines([all_names[i] + '\n' for i in ind])
        f.close()

        cmd = '%s %s %s' % (
            self._paths['compute_ap'],
            os.path.join(self._paths['groundtruth'], query_name), temp_filename)
        ap = os.popen(cmd).read()

        # Delete temporary file.
        os.remove(temp_filename)
        return float(ap.strip())


def main():
    """Main function of this program."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', type=str, required=True,
                        help='Dataset to evaluate.')
    parser.add_argument('--via', dest='via', type=str, required=True,
                        help='Dataset to assistant PCA whitening.')
    parser.add_argument('--crop', dest='crop', type=bool, required=True,
                        help='Whether to use cropped query.')
    parser.add_argument('--detectors', dest='n_detectors', type=int,
                        required=True, help='Number part detectors.')
    parser.add_argument('--dim', dest='dim', type=int, required=True,
                        help='Dimension of the final retrieval descriptor.')
    parser.add_argument('--whiten', dest='whiten', type=bool, required=True,
                        help='Whether to use whitening.')
    args = parser.parse_args()
    if args.test not in ['oxford', 'paris']:
        raise AttributeError('--test parameter must be oxford/paris.')
    if args.via not in ['oxford', 'paris']:
        raise AttributeError('--via parameter must be oxford/paris.')
    if args.n_detectors <= 0:
        raise AttributeError('--detectors parameter must >0.')
    if args.dim <= 0:
        raise AttributeError('--dim parameter must >0.')

    test_image_root = os.path.join('/data/zhangh/data/', args.test)
    via_image_root = os.path.join('/data/zhangh/data/', args.via)
    project_root = '/data/zhangh/project/pwa/'
    paths = {
        'groundtruth': os.path.join(test_image_root, 'groundtruth/'),
        'test_all_pool5': os.path.join(test_image_root, 'pool5/all/'),
        'test_crop_pool5': os.path.join(test_image_root, 'pool5/crop/'),
        'test_full_pool5': os.path.join(test_image_root, 'pool5/full/'),
        'via_all_pool5': os.path.join(via_image_root, 'pool5/all/'),
        'compute_ap': os.path.join(project_root, 'lib/compute_ap'),
    }
    for k in paths:
        if k != 'compute_ap':
            assert os.path.isdir(paths[k])
        else:
            assert os.path.isfile(paths[k])

    evaluate_manager = EvaluateManager(paths, n_detectors=args.n_detectors)
    evaluate_manager.fitPca(dim=args.dim, whiten=args.whiten)
    evaluate_manager.evaluate(args.crop)


if __name__ == '__main__':
    main()
