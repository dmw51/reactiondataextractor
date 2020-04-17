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


class Reaction(BaseReactionClass):
    @classmethod
    def from_reaction_steps(cls, steps):
        """
        Given reactions steps forming the reaction, return a Reaction object with steps ordered in a list

        :param [ReactionStep,...] steps: list of all involved reaction steps
        :return: Reactio object - an iterator over the reaction steps
        """
        ordered_steps = []
        first_step = [step for step in steps if step.first][0]
        # TODO: Exception handling for above
        ordered_steps.append(first_step)

        compare_to_step = first_step
        changes = True
        while changes:
            changes = False
            for step in steps:

                if set(step.reactants) == set(compare_to_step.products):
                    changes = True
                    ordered_steps.append(step)
                    compare_to_step = step

        return Reaction(steps=ordered_steps)

    def __init__(self, steps):
        self.steps = steps


class ChemicalStructure(BaseReactionClass):
    """
    This is a base class for chemical structures species found in diagrams (e.g. reactants and products)
    """
    def __init__(self, connected_components):
        self.connected_components = connected_components
        self.smiles = None
        self.label = None

    def __iter__(self):
        return iter(self.connected_components)


class ReactionStep(BaseReactionClass):
    """
    This class describes elementary steps in a reaction.
    """

    def __init__(self, arrow, reactants, products, conditions, first):
        self.arrow = arrow,
        self.reactants = reactants
        self.products = products
        self.conditions = conditions
        self._first = first   #  internal flag to indicate first step in a reaction

    @property
    def first(self):
        return self._first


class Conditions:
    """
    This class describes conditions region and associated text
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