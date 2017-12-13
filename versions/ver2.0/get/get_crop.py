#!/usr/bin/env python
"""Exatract cropped query images for Oxford and Paris dataset.

This program is modified from CroW:
    https://github.com/yahoo/crow/
"""


from __future__ import division, print_function


__all__ = ['CropManager']
__author__ = 'Hao Zhang'
__copyright__ = 'Copyright @2017 LAMDA'
__date__ = '2017-09-26'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-09-26'
__version__ = '2.0'


import argparse
import glob
import itertools
import os
import sys
if sys.version[0] == '2':
    filter = itertools.ifilter
    input = raw_input
    map = itertools.imap
    range = xrange
    zip = itertools.izip

import PIL.Image


class CropManager(object):
    """Exatract Oxford cropped query images.

    Attributes:
        _dataset, str: Name of the dataset, either 'oxford' or 'paris'.
        _paths, dict of (str, str): Image data and feature paths.
    """
    def __init__(self, dataset, paths):
        self._dataset = dataset
        self._paths = paths

    def getCrop(self):
        """Extract cropped query images from Oxford or Paris dataset.

        The cropped query images are save in path self._paths['crop_image'].
        """
        print('Exact cropped queries from all.')
        for f in glob.iglob(os.path.join(self._paths['groundtruth'],
                                         '*_query.txt')):
            query_name = os.path.splitext(
                os.path.basename(f))[0].replace('_query', '')
            image_name, x, y, w, h = open(f).read().strip().split(' ')
            if self._dataset == 'oxford':
                image_name = image_name.replace('oxc1_', '')

            image = PIL.Image.open(os.path.join(self._paths['all_image'],
                                                '%s.jpg' % image_name))
            # Fetch cropped query region from the image.
            x, y, w, h = map(float, (x, y, w, h))
            box = map(lambda d: int(round(d)), (x, y, x + w, y + h))
            image = image.crop(box)
            image.save('%s.png' % os.path.join(self._paths['crop_image'],
                                               query_name))


def main():
    """Main function of the program,"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='Dataset to extract pool5 features.')
    args = parser.parse_args()
    if args.dataset not in ['oxford', 'paris']:
        raise AttributeError('--dataset parameter must be oxford/paris.')

    image_root = os.path.join('/data/zhangh/data/', args.dataset)
    paths = {
        'all_image': os.path.join(image_root, 'image/all/'),
        'crop_image': os.path.join(image_root, 'image/crop/'),
        'groundtruth': os.path.join(image_root, 'groundtruth/'),
    }
    for k in paths:
        assert os.path.isdir(paths[k])

    manager = CropManager(args.dataset, paths)
    manager.getCrop()


if __name__ == '__main__':
    main()
