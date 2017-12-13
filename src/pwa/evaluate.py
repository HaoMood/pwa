#!/usr/bin/env python
"""Evaluate the result.

This program is modified from CroW:
    https://github.com/yahoo/crow/

The compute_ap executable file provied by the Oxford5k dataset:
    http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
To compute the average precision for a ranked list of query christ_church_1,
run:
    ./compute_ap christ_church_1 ranked_list.txt
"""


import os
import tempfile

import numpy as np
import sklearn.decomposition
import sklearn.metrics.pairwise
import sklearn.neighbors
import sklearn.preprocessing


__all__ = ['EvaluateManager']
__author__ = 'Hao Zhang'
__date__ = '2017-11-21'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-11-03'
__version__ = '3.0'


class EvaluateManager(object):
    """Manager class to compute mAP.

    Attributes:
        _dim, int: Dimension of the final retrieval descriptor.
        _num_channels, int: Number channels to compute Gram.
        _paths, dict of (str, str): Feature paths.
        _pca_model: sklearn.decomposition.PCA model for final PCA whitening.
        _selected_channels, np.ndarray: Selected channel indices.
    """
    def __init__(self, paths, dim):
        print('Initiate.')
        self._paths = paths
        self._dim = dim
        self._pca_model = None
        self._selected_channels = np.load(os.path.join(self._paths['channels'],
                                                       'channels.npy'))
        self._num_channels = self._selected_channels.size

    def fitPca(self):
        """Compute PCA whitening paramters from the via dataset.

        Args:
            dim, int: Dimension of the final retrieval descriptor.
        """
        print('Load via dataset features.')
        # Load all Gram matrices.
        via_all_descriptors, _ = self._loadFeature('via_all_conv')

        print('Fit PCA whitening paramters from via dataset.')
        # l2-normalize.
        via_all_descriptors = np.vstack(via_all_descriptors)
        sklearn.preprocessing.normalize(via_all_descriptors, copy=False)
        self._pca_model = sklearn.decomposition.PCA(n_components=self._dim,
                                                    whiten=True)
        self._pca_model.fit(via_all_descriptors)

    def evaluate(self):
        """Evaluate the retrieval results."""
        print('Load test dataset features.')
        # Load all Gram matrices.
        test_all_descriptors, test_all_names = self._loadFeature(
            'test_all_conv')
        test_crop_descriptors, test_crop_names = self._loadFeature(
            'test_crop_conv')

        # l2-normalize, PCA whitening, and l2-normalize again.
        test_all_descriptors, test_crop_descriptors = self._normalization(
            test_all_descriptors, test_crop_descriptors)
        assert test_all_descriptors.shape[1] == test_crop_descriptors.shape[1]

        # Iterate queries, process them, rank results, and evaluate mAP.
        print('Evaluate the results.')
        all_ap = []
        for i in range(len(test_crop_names)):
            knn_model = sklearn.neighbors.NearestNeighbors(
                n_neighbors=len(test_all_descriptors))
            knn_model.fit(test_all_descriptors)
            _, ind = knn_model.kneighbors(
                test_crop_descriptors[i].reshape(1, -1))
            ap = self._getAP(ind[0], test_crop_names[i], test_all_names)
            all_ap.append(ap)
        m_ap = np.mean(np.array(all_ap))
        print('mAP is', m_ap)

    def _loadFeature(self, conv_path):
        """Load and process conv features into weighted-pooled descriptors.

        This is a helper function of fitPca() and evaluate().

        Args:
            conv_path, str: Path of .npy conv features.

        Return:
            descriptor_list, list of np.ndarray: List of gram matrices.
            name_list, list of str: List of image names without extensions.
        """
        name_list = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in os.listdir(self._paths[conv_path])
            if os.path.isfile(os.path.join(self._paths[conv_path], f))])
        m = len(name_list)
        descriptor_list = []
        for i, name_i in enumerate(name_list):
            if i % 200 == 0:
                print('Processing %d/%d' % (i, m))
            # Load conv feature.
            conv_i = np.load('%s.npy' %
                             os.path.join(self._paths[conv_path], name_i))
            D, H, W = conv_i.shape
            assert D == 512

            pooled_descriptor = []
            for channel_k in self._selected_channels:
                weight_k = conv_i[channel_k, :, :]
                weight_k = np.reshape(weight_k, (1, H * W))
                sklearn.preprocessing.normalize(weight_k, copy=False)
                weight_k = np.sqrt(weight_k)
                weight_k = np.reshape(weight_k, (1, H, W))
                descriptor_k = np.sum(weight_k * conv_i, axis=(1, 2))
                pooled_descriptor.append(descriptor_k[np.newaxis, :])
            pooled_descriptor = np.hstack(pooled_descriptor)
            assert pooled_descriptor.shape == (1, 512 * self._num_channels)

            descriptor_list.append(pooled_descriptor)
        return descriptor_list, name_list

    def _normalization(self, test_all_descriptors, test_crop_descriptors):
        """l2 normalize, PCA whitening, and l2 normalize again for test all and
        cropped query descriptors.

        This is a helper function of evaluate().

        Args:
            test_all_descriptors, np.ndarray: Before normalize.
            test_crop_descriptors, np.ndarray: Before normalize.

        Return
            test_all_descriptors, np.ndarray: After normalize.
            test_crop_descriptors, np.ndarray: After normalize.
        """
        test_all_descriptors = np.vstack(test_all_descriptors)
        test_crop_descriptors = np.vstack(test_crop_descriptors)

        sklearn.preprocessing.normalize(test_all_descriptors, copy=False)
        sklearn.preprocessing.normalize(test_crop_descriptors, copy=False)

        test_all_descriptors = self._pca_model.transform(test_all_descriptors)
        test_crop_descriptors = self._pca_model.transform(test_crop_descriptors)

        sklearn.preprocessing.normalize(test_all_descriptors, copy=False)
        sklearn.preprocessing.normalize(test_crop_descriptors, copy=False)
        assert test_all_descriptors.shape[1] == self._dim
        assert test_crop_descriptors.shape[1] == self._dim

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', type=str, required=True,
                        help='Dataset to evaluate.')
    parser.add_argument('--via', dest='via', type=str, required=True,
                        help='Dataset to assistant PCA whitening.')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='Model to extract features.')
    parser.add_argument('--dim', dest='dim', type=int, required=True,
                        help='Dimension of the final retrieval descriptor.')
    args = parser.parse_args()
    if args.test not in ['oxford5k', 'paris6k']:
        raise AttributeError('--test parameter must be oxford5k/paris6k.')
    if args.via not in ['oxford5k', 'paris6k']:
        raise AttributeError('--via parameter must be Oxford5k/paris6k.')
    if args.model not in ['vgg16', 'vgg19']:
        raise AttributeError('--model parameter must be vgg16/vgg19.')
    if args.dim <= 0:
        raise AttributeError('--dim parameter must >0.')

    project_root = os.popen('pwd').read().strip()
    test_data_root = os.path.join(os.path.join(project_root, 'data'), args.test)
    via_data_root = os.path.join(os.path.join(project_root, 'data'), args.via)
    paths = {
        'groundtruth': os.path.join(test_data_root, 'groundtruth/'),
        'test_all_conv': os.path.join(os.path.join(os.path.join(
            test_data_root, 'conv'), args.model), 'all/'),
        'test_crop_conv': os.path.join(os.path.join(os.path.join(
            test_data_root, 'conv'), args.model), 'crop/'),
        'via_all_conv': os.path.join(os.path.join(os.path.join(
            via_data_root, 'conv'), args.model), 'all/'),
        'compute_ap': os.path.join(project_root, 'lib/compute_ap'),
        # 'channels': test_data_root,
        'channels': via_data_root,
    }
    for k in paths:
        if k != 'compute_ap':
            assert os.path.isdir(paths[k])
        else:
            assert os.path.isfile(paths[k])

    evaluate_manager = EvaluateManager(paths, args.dim)
    evaluate_manager.fitPca()
    evaluate_manager.evaluate()


if __name__ == '__main__':
    main()
