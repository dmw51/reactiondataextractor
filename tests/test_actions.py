import pytest

import numpy as np
import matplotlib.pyplot as plt

from models.segments import Rect
from models.reaction import ReactionStep, Reactant, Product

from actions import match_function_and_smiles

def test_match_function_and_smiles():
    smiles = ([(['1c-Br2 92% yield, >99%ee'], 'Brc1ccc2c(c3c4ccc(Br)cc4ccc3O)c(O)ccc2c1'),
               (['1c (05 mmol)'],'c1ccc2c(c3c4ccccc4ccc3O)c(O)ccc2c1'),
               (['NBS (2 6 equiv)'], 'C1(=O)N(C(=O)CC1)Br')],
              [Rect(left=711, right=948, top=0, bottom=200),
               Rect(left=0, right=186, top=6, bottom=193),
               Rect(left=260, right=376, top=29, bottom=176)])


    reactants = [Reactant(Rect (0, 186, 6, 193)), Reactant(Rect(260, 376, 29, 176))]
    products = [Product(Rect(711, 948, 0, 200))]
    reaction_step = ReactionStep(None, reactants, products, None)
    cc1 = smiles[1][0]


    match_function_and_smiles(reaction_step, smiles)

    assert reactants[0].smiles == 'c1ccc2c(c3c4ccccc4ccc3O)c(O)ccc2c1'
    assert reactants[0].label == ['1c (05 mmol)']

    assert reactants[1].smiles == 'C1(=O)N(C(=O)CC1)Br'
    assert reactants[1].label == ['NBS (2 6 equiv)']

    assert products[0].smiles == 'Brc1ccc2c(c3c4ccc(Br)cc4ccc3O)c(O)ccc2c1'
    assert products[0].label == ['1c-Br2 92% yield, >99%ee']


if __name__ == '__main__':
    test_match_function_and_smiles()
