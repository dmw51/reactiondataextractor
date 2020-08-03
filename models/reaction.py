from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import copy
from itertools import product, tee
import numpy as np
from enum import Enum

from utils.processing import remove_small_fully_contained

log = logging.getLogger(__name__)





class BaseReactionClass(object):
    """
    This is a base reaction class placeholder
    """


# class Reaction(BaseReactionClass):
#     @classmethod
#     def from_reaction_steps(cls, steps):
#         """
#         Given reactions steps forming the reaction, return a Reaction object with steps ordered in a list
#
#         :param [ReactionStep,...] steps: list of all involved reaction steps
#         :return: Reactio object - an iterator over the reaction steps
#         """
#         ordered_steps = []
#         try:
#             first_step = [step for step in steps if step.first][0]
#         except IndexError:
#             log.warning('First step of the reaction was not found.')
#             return None
#
#         # TODO: Exception handling for above
#         ordered_steps.append(first_step)
#
#         compare_to_step = first_step
#         changes = True
#         while changes:
#             changes = False
#             for step in steps:
#                 if set(step.reactants) == set(compare_to_step.products):
#                     changes = True
#                     ordered_steps.append(step)
#                     compare_to_step = step
#
#         return Reaction(steps=ordered_steps)
#
#     def __init__(self, steps):
#         self.steps = steps


class ChemicalStructure(BaseReactionClass):
    """
    This is a base class for chemical structures species found in diagrams (e.g. reactants and products)
    """
    def __init__(self, panel):
        self.panel = panel
        self.smiles = None
        self.label = None

    # def __iter__(self):
    #     return iter(self.panel)

    def __eq__(self, other):
        if isinstance(other, ChemicalStructure):   # Only compare exact same types
            return self.panel == other.panel and self.label == other.label and self.smiles == other.smiles

    def __hash__(self):
        return hash(self.panel)

    def __repr__(self):
        return f'{self.__class__.__name__}(connected_component={self.panel}, smiles={self.smiles}, label={self.label})'

    def __str__(self):
        return f'{self.smiles}, label: {self.smiles}'

class ReactionStep(BaseReactionClass):
    """
    This class describes elementary steps in a reaction.
    """

    def __init__(self, reactants, products, conditions):
        self.reactants = frozenset(reactants)
        self.products = frozenset(products)
        self.conditions = conditions

    def __eq__(self, other):
        return (self.arrow == other.arrow and self.reactants == other.reactants and self.products == other.products and
                self.conditions == other.conditions)

    def __repr__(self):
        return f'ReactionStep({self.reactants},{self.products},{self.conditions})'

    def __str__(self):
        return '+'.join(self.reactants)+'-->'+'+'.join(self.products)

    def __hash__(self):
        return hash(self.arrow) + hash(self.conditions)

    def __iter__(self):
        return iter ((self.reactants, self.products))

    @property
    def first(self):
        return self._first




class Conditions:
    """
    This class describes conditions region and associated text
    """

    def __init__(self, text_lines, conditions_dct, arrow):
        self.arrow = arrow
        self.text_lines = text_lines
        self.conditions_dct = conditions_dct
        self.text_lines.sort(key=lambda textline: textline.panel.top)

    def __repr__(self):
        return f'Conditions({self.text_lines}, {self.conditions_dct}, {self.arrow})'

    def __str__(self):
        return "\n".join(f'{key} : {value}' for key, value in self.conditions_dct.items() if value)

    def __eq__(self, other):
        if other.__class__ == self.__class__:
            return self.conditions_dct == other.conditions_dct

    def __hash__(self):
        return hash(sum(hash(line) for line in self.text_lines))

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