from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import numpy as np

from .segments import Rect

log = logging.getLogger(__name__)


class BaseReactionClass(object):
    """
    This is a base reaction class placeholder
    """


class ReactionStep(BaseReactionClass):
    """
    This class describes elementary steps in a reaction.
    """

    def __init__(self, arrow_type, reactants, products, conditions):
        self.arrow_type = arrow_type
        self.reactants = reactants
        self.products = products
        self.conditions = conditions

class Conditions(BaseReactionClass):
    """
    This class describes conditions
    """

class Reactant(BaseReactionClass):
    """
    This class describes reactants
    """

class Intermediate(BaseReactionClass):
    """
    This class describes reaction intermediates
    """

class Product(BaseReactionClass):
    """
    This class describes final reaction products
    """