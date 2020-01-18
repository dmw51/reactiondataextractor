from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import copy
from itertools import product, tee
import numpy as np


from utils.processing import merge_overlapping, remove_small_fully_contained

log = logging.getLogger(__name__)


class BaseReactionClass(object):
    """
    This is a base reaction class placeholder
    """


class ReactionElement(BaseReactionClass):
    """
    This is a base class for reaction elements found in diagrams (e.g. reactants,conditions and products)
    """
    def __init__(self, connected_components):
        self.connected_components = remove_small_fully_contained(connected_components)

        # while True:
        #     merged = []
        #     for cc1 in previous_container:
        #         for cc2 in previous_container:
        #             if cc1.overlaps(cc2) and cc1 != cc2:
        #                 print('cc1 in m list', cc1 in merging_list)
        #                 print(cc1)
        #                 merging_list.remove(cc1)
        #                 merging_list.remove(cc2)
        #                 merged = merge_rect(cc1, cc2)
        #                 merging_list.add(merged)
        #                 print('merged in mlist:, ', merged in merging_list)
        #                 cc1 = merged

        #     if len(merging_list) == len(previous_container):
        #         print('merging complete')
        #         break
        #     previous_container = merging_list
        #
        # print('---')
        # self.connected_components = merged

    def __iter__(self):
        return iter(self.connected_components)


class ReactionStep(BaseReactionClass):
    """
    This class describes elementary steps in a reaction.
    """

    def __init__(self, arrow, reactants, products, conditions):
        self.arrow = arrow,
        self.reactants = reactants
        self.products = products
        self.conditions = conditions


class Conditions(ReactionElement):
    """
    This class describes conditions
    """

    def __init__(self, connected_components):
        super(Conditions, self).__init__(connected_components)

class Reactant(ReactionElement):
    """
    This class describes reactants
    """
    def __init__(self, connected_components):
        super(Reactant, self).__init__(connected_components)

class Intermediate(ReactionElement):
    """
    This class describes reaction intermediates
    """
    def __init__(self, connected_components):
        super(Intermediate, self).__init__(connected_components)

class Product(ReactionElement):
    """
    This class describes final reaction products
    """
    def __init__(self, connected_components):
        super(Product, self).__init__(connected_components)