#!/usr/bin/env python
"""Exatract Oxford cropped query images.

This program is modified from CroW:
    https://github.com/yahoo/crow/
"""


from __future__ import division, print_function


__all__ = ['OxfordManager']
__author__ = 'Hao Zhang'
__copyright__ = 'Copyright @2017 LAMDA'
__date__ = '2017-09-17'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-09-17'
__version__ = '1.1'


import glob
import os
import sys
if sys.version[0] == '2':
    input = raw_input

import PIL.Image


class OxfordManager(object):
    """Exatract Oxford cropped query images.

    Attributes:
        _paths, dict of (str, str): Oxford data and feature paths.
    """
    def __init__(self, paths):
        self._paths = paths

    def getCrop(self):
        """Extract cropped query images for Oxford.

        The cropped query images are save in path self._paths['crop'].
        """
        print('Exact cropped queries from all.')
        for f in glob.iglob(os.path.join(self._paths['groundtruth'],
                                         '*_query.txt')):
            query_name = os.path.splitext(
                os.path.basename(f))[0].replace('_query', '')
            image_name, x, y, w, h = open(f).read().strip().split(' ')
            image_name = image_name.replace('oxc1_', '')

            image = PIL.Image.open(os.path.join(self._paths['original'],
                                                image_name + '.jpg'))
            x, y, w, h = map(float, (x, y, w, h))
            box = map(lambda d: int(round(d)), (x, y, x + w, y + h))
            # Return a rectangular region from the image.
            image = image.crop(box)
            image.save(os.path.join(self._paths['crop'], query_name)
                       + '.jpg')


def main():
    """Main function of the program,"""
    paths = {
        'original': '/data/zhangh/data/oxbuild/images/',
        'crop': '/data/zhangh/data/oxbuild/cropped-queries/',
        'groundtruth': '/data/zhangh/data/oxbuild/groundtruth/',
    }
    manager = OxfordManager(paths)
    manager.getCrop()


if __name__ == '__main__':
    main()
