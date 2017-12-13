#!/usr/bin/env python
"""Find features for query images and evaluate the result.

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
__date__ = '2017-09-15'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-09-15'
__version__ = '1.0'


import glob
import os
import tempfile

import numpy as np
import sklearn.neighbors


class EvaluateManager(object):
    """Manager class to compute mAP.

    Attributes:
        _paths (dict of (str, str)): Oxford data and feature paths.
    """
    def __init__(self, paths):
        self._paths = paths

    def extractQuery(self):
        """Extract descriptors corresponding to query images.

        Note that we use full image instead of cropped image for query.
        """
        print('Extract query descriptors from all.')
        for f in glob.iglob(os.path.join(self._paths['groundtruth'],
                                         '*_query.txt')):
            query_name = os.path.splitext(
                os.path.basename(f))[0].replace('_query', '')
            image_name, _, _, _, _ = open(f).read().strip().split(' ')
            image_name = image_name.replace('oxc1_', '')
            os.system('cp %s.npy %s.npy' %
                      (os.path.join(self._paths['descriptor_all'], image_name),
                       os.path.join(self._paths['descriptor_query'],
                                    query_name)))

    def evaluate(self):
        """Evaluate the retrieval results."""
        print('Load all descriptors.')
        all_features, all_names = self._loadFeature('descriptor_all')
        all_features = np.vstack(all_features)
        print('Load query descriptors.')
        query_features, query_names = self._loadFeature('descriptor_query')
        query_features = np.vstack(query_features)
        number_all = len(all_names)
        assert all_features.shape == (number_all, 4096)
        number_queries = len(query_names)
        assert query_features.shape == (number_queries, 4096)

        # Iterate queries, process them, rank results, and evaluate mAP.
        print('Compute AP for each query.')
        all_ap = []
        for i in xrange(len(query_names)):
            if i % 10 == 0:
                print('Process %d/%d' % (i, number_queries))
            # NearestNeighbors(n_neighbors=5, algorithm='auto', p=2)
            # - n_neighbors: Number of neighbors to use by default for
            #   kneighbors queries.
            # - p: Parameter for the Minkowski metric, l2 distance by default.
            # Methods
            # - fit(X): Fit the model using X as training data.
            # - kneighbors(X=None, n_neighbors=None, return_distance=True):
            #   Finds the kNN of a point. Return indices of and distances to
            #   the neighbors of each point.
            knn_model = sklearn.neighbors.NearestNeighbors(
                n_neighbors=number_all)
            knn_model.fit(all_features)
            _, ind = knn_model.kneighbors(query_features[i].reshape(1, -1))
            ap = self._getAP(ind[0], query_names[i], all_names)
            all_ap.append(ap)
        m_ap = np.array(all_ap).mean()
        print('mAP is', m_ap)

    def _loadFeature(self, path):
        """Load all/query features into a list and also return a list of the
        corresponding filenames without the file extension.

        This is a helper function of evaluate().

        Args:
            path (str): Indicate load path
        """
        all_names = [
            f[:-4] for f in os.listdir(self._paths[path])
            if os.path.isfile(os.path.join(self._paths[path], f))]
        all_features = []
        for name_i in all_names:
            descriptor_i = np.load(os.path.join(self._paths[path], name_i) +
                                   '.npy')
            assert descriptor_i.shape == (4096,)
            all_features.append(descriptor_i[np.newaxis, :])
        return all_features, all_names

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

        cmd = ('%scompute_ap %s %s' %
               (self._paths['compute_ap'],
                os.path.join(self._paths['groundtruth'], query_name),
                temp_filename))
        ap = os.popen(cmd).read()

        # Delete temporary file.
        os.remove(temp_filename)
        return float(ap.strip())


def main():
    """Main function of this program."""
    project_root = '/data/zhangh/project/pwa/'
    paths = {
        'groundtruth': '/data/zhangh/data/oxbuild/groundtruth/',
        'descriptor_all': os.path.join(project_root, 'data/descriptor/all/'),
        'descriptor_query': os.path.join(project_root,'data/descriptor/query/'),
        'compute_ap': os.path.join(project_root, 'lib/'),
    }
    evaluate_manager = EvaluateManager(paths)
    # evaluate_manager.extractQuery()
    evaluate_manager.evaluate()


if __name__ == '__main__':
    main()
