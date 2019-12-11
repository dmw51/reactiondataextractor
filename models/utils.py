from collections import namedtuple

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

log = logging.getLogger((__name__))

Point = namedtuple('Point','row col')
#Line class?

Line = namedtuple('Line','pixels')