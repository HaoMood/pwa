#!/usr/bin/env python
"""Select channels for PWA.

This program is modified from CroW:
    https://github.com/yahoo/crow/
"""


import os

import numpy as np


__all__ = ['PWA']
__author__ = 'Hao Zhang'
__date__ = '2017-11-21'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-11-03'
__version__ = '3.0'


class PWA(object):
    """Manager class to select channels.

    Attributes:
        _num_channels, int: Number channels to for PWA.
        _paths, dict of (str, str): Feature paths.
    """
    def __init__(self, paths, channels):
        print('Initiate.')
        self._paths = paths
        self._num_channels = channels

    def selectChannels(self):
        """Compute variance for each dimension of the pooled descriptor and
        select part detectors.

        This is a helper function of fitPca().
        """
        print('Load via dataset features.')
        # Load all Gram matrices.
        all_descriptors = self._loadFeature('all_conv')

        print('Select channels from pooled descriptors.')
        # Compute variance for each channel.
        channel_var = np.var(np.vstack(all_descriptors), axis=0)
        assert channel_var.shape == (512,)
        selected_channels = np.argsort(channel_var)[::-1][:self._num_channels]
        assert selected_channels.shape == (self._num_channels,)
        np.save(os.path.join(self._paths['channels'], 'channels'),
                selected_channels)

    def _loadFeature(self, conv_path):
        """Load and process conv features into pooled descriptors.

        This is a helper function of selectChannels().

        Args:
            conv_path, str: Path of .npy conv features.

        Return:
            descriptor_list, list of np.ndarray: List of gram matrices.
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

            pooled_descriptor = np.sum(conv_i, axis=(1, 2))
            assert pooled_descriptor.shape == (512,)
            descriptor_list.append(pooled_descriptor[np.newaxis, :])
        return descriptor_list


def main():
    """Main function of this program."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='Dataset to select channels.')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='Model to extract features.')
    parser.add_argument('--channels', dest='channels', type=int, required=True,
                        help='Number channels to select.')
    args = parser.parse_args()
    if args.dataset not in ['oxford5k', 'paris6k']:
        raise AttributeError('--dataset parameter must be oxford5k/paris6k.')
    if args.model not in ['vgg16', 'vgg19']:
        raise AttributeError('--model parameter must be vgg16/vgg19.')
    if not 0 < args.channels <= 512:
        raise AttributeError('--channels parameter must in range (0, 512].')

    project_root = os.popen('pwd').read().strip()
    data_root = os.path.join(os.path.join(project_root, 'data'), args.dataset)
    paths = {
        'all_conv': os.path.join(os.path.join(os.path.join(
            data_root, 'conv'), args.model), 'all/'),
        'channels': data_root,
    }
    for k in paths:
        assert os.path.isdir(paths[k])

    pwa = PWA(paths, args.channels)
    pwa.selectChannels()


if __name__ == '__main__':
    main()
