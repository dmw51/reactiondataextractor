from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

from .models.segments import Figure
from skimage import io
def imread(path_):
    """
    This function takes in an image and returns binarised Figure object
    :param string path_: path to a file
    :return: Figure
    """


