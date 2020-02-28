from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import copy
from itertools import product, tee
import numpy as np


from utils.processing import remove_small_fully_contained

log = logging.getLogger(__name__)


class BaseReactionClass(object):
    """
    This is a base reaction class placeholder
    """


class ChemicalStructure(BaseReactionClass):
    """
    This is a base class for chemical structures species found in diagrams (e.g. reactants and products)
    """
    def __init__(self, connected_components):
        self.connected_components = connected_components
            #remove_small_fully_contained(connected_components)

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


class Conditions:
    """
    This class describes conditions
    """

    def __init__(self, text_lines, conditions_dct):
        self.text_lines = text_lines
        self.conditions_dct = conditions_dct

    @property
    def co_reactants(self):
        return self.conditions_dct['co-reactants']

    @property
    def catalysts(self):
        return self.conditions_dct['catalysts']

    @property
    def other_species(self):
        return self.conditions_dct['other species']

    @property
    def temperature(self):
        return self.conditions_dct['temperature']

    @property
    def time(self):
        return self.conditions_dct['time']

    @property
    def pressure(self):
        return self.conditions_dct['pressure']

    @property
    def yield_(self):
        return self.conditions_dct['yield']


class Reactant(ChemicalStructure):
    """
    This class describes reactants
    """
    def __init__(self, connected_components):
        super(Reactant, self).__init__(connected_components)

class Intermediate(ChemicalStructure):
    """
    This class describes reaction intermediates
    """
    def __init__(self, connected_components):
        super(Intermediate, self).__init__(connected_components)

class Product(ChemicalStructure):
    """
    This class describes final reaction products
    """
    def __init__(self, connected_components):
        super(Product, self).__init__(connected_components)