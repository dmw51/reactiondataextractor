from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

from models.segments import Panel

log = logging.getLogger(__name__)


class BaseReactionClass(object):
    """
    This is a base reaction class placeholder
    """


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
        return f'{self.__class__.__name__}(panel={self.panel}, smiles={self.smiles}, label={self.label})'

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
        return (self.reactants == other.reactants and self.products == other.products and
                self.conditions == other.conditions)

    def __repr__(self):
        return f'ReactionStep({self.reactants},{self.products},{self.conditions})'

    def __str__(self):
        return '+'.join(self.reactants)+'-->'+'+'.join(self.products)

    def __hash__(self):
        all_species = [species for group in iter(self) for species in group]
        species_hash = sum([hash(species) for species in all_species])
        return hash(self.conditions) + species_hash

    def __iter__(self):
        return iter ((self.reactants, self.products))

    def match_function_and_smiles(self, csr_output):
        """
        Matches the resolved smiles from chemschematicresolver with roles (reactant, product) found by the segmentation
        algorithm.

        :param [[smile], [ccs]] csr_output: list of lists containing structures converted into SMILES format and recognised
         labels, and connected components depicting the structures in an image
        :return: bool True if matching successful else False
        """
        smile_panel_pairs = list(zip(*csr_output))
        for reactant in self.reactants:
            matching_record = [recognised for recognised, diag in smile_panel_pairs
                               if Panel(diag.left, diag.right, diag.top, diag.bottom) == reactant.panel]
            # This __eq__ is based on a flaw in csr - cc is of type `Label`, but inherits from Rect
            if matching_record:
                matching_record = matching_record[0]
                reactant.label = matching_record[0]
                reactant.smiles = matching_record[1]
            else:
                log.warning('No SMILES match was found for a reactant structure')

        for product in self.products:

            matching_record = [recognised for recognised, diag in smile_panel_pairs
                               if diag == product.panel]
            # This __eq__ is based on flaw in csr - cc is of type `Diagram`, but inherits from Rect
            if matching_record:
                matching_record = matching_record[0]
                product.label = matching_record[0]
                product.smiles = matching_record[1]
            else:
                log.warning('No SMILES match was found for a product structure')

        if all([reactant.smiles for reactant in self.reactants]) and \
            all ([product.smiles for product in self.products]):
            print('all structures were translated to SMILES')
            return True
        else:
            print('No SMILES were found for some structures - extraction unsuccessful')
            return False


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

        else:
            return False
    #
    # def __getitem__(self, item):
    #     return self.conditions_dct[item]

    def __hash__(self):
        return hash(sum(hash(line) for line in self.text_lines))

    @property
    def anchor(self):
        a_pixels = self.arrow.pixels
        return a_pixels[len(a_pixels)//2]
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




#