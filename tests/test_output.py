from collections import namedtuple
import pytest

from models.reaction import ReactionStep, Reactant, Product
from models.segments import Panel
from models.output import ReactionScheme

class ConditionsDummy:

    def __init__(self, dct):
        self.dct = dct

    def __str__(self):
        return str(self.dct)

class SpeciesDummy:

    def __init__(self, smiles, label):
        self.smiles = smiles
        self.label = label


    def __iter__(self):
        return iter([self.smiles, self.label])

conditions = [ConditionsDummy({i: i}) for i in range(10)]

species = [SpeciesDummy(letter, label) for letter, label in zip(list('abcdefghijkl'), range(10))]


def test_create_graph_single_path():
    steps = [ReactionStep([r], [p], c) for r, p, c in zip(species[:3], species[1:4], conditions)][:2]
    true_graph = {group: [] for step in steps for group in (step.reactants, step.conditions, step.products)}
    for step in steps:
        true_graph[step.reactants] = [step.conditions]
        true_graph[step.conditions] = [step.products]

    scheme = ReactionScheme(steps)
    assert scheme.graph == true_graph


def test_create_graph_multiple_paths():
    steps1 = [ReactionStep([r], [p], c) for r, p, c in zip(species[:3], species[1:4], conditions)][:2]
    steps2 = [ReactionStep([r], [p], c) for r, p, c in zip(species[3:6], species[4:7], conditions[2:4])][:2]
    steps = steps1 + steps2

    true_graph = {group: [] for step in steps for group in (step.reactants, step.conditions, step.products)}
    for step in steps:
        true_graph[step.reactants].append(step.conditions)
        true_graph[step.conditions].append(step.products)

    scheme = ReactionScheme(steps)

    assert scheme.graph == true_graph

def test_set_start_end_nodes_single_path():
    steps = [ReactionStep([r], [p], c) for r, p, c in zip(species[:3], species[1:4], conditions)][:2]
    for step in steps:
        [setattr(s, '__class__', Reactant) for s in step.reactants]
        [setattr(s, 'panel', (0, 0, 0, 0)) for s in step.reactants]

        [setattr(s, '__class__', Product) for s in step.products]
        [setattr(s, 'panel', (0, 0, 0, 0)) for s in step.products]

    scheme = ReactionScheme(steps)
    assert scheme._start == [steps[0].reactants]
    assert scheme._end == [(steps[-1].products)]


def test_set_start_end_nodes_separate_paths():
    steps1 = [ReactionStep([r], [p], c) for r, p, c in zip(species[:3], species[1:4], conditions)][:2]
    steps2 = [ReactionStep([r], [p], c) for r, p, c in zip(species[3:6], species[4:7], conditions[2:4])][:2]
    steps = steps1 + steps2
    for step in steps:
        [setattr(s, '__class__', Reactant) for s in step.reactants]
        [setattr(s, 'panel', (0, 0, 0, 0)) for s in step.reactants]

        [setattr(s, '__class__', Product) for s in step.products]
        [setattr(s, 'panel', (0, 0, 0, 0)) for s in step.products]

    scheme = ReactionScheme(steps)
    assert  scheme._start == [steps1[0].reactants, steps2[0].reactants]
    assert scheme._end == [steps1[-1].products, steps2[-1].products]


def test_set_start_end_nodes_multiple_products():
    steps1 = [ReactionStep([r], [p], c) for r, p, c in zip(species[:3], species[1:4], conditions)][:2]
    branch_group = steps1[-1].products
    branched_steps = [ReactionStep(branch_group, [p], c) for p, c in zip(species[3:6], conditions[2:4])][:3]

    steps = steps1+branched_steps

    for step in steps:
        [setattr(s, '__class__', Reactant) for s in step.reactants]
        [setattr(s, 'panel', (0, 0, 0, 0)) for s in step.reactants]

        [setattr(s, '__class__', Product) for s in step.products]
        [setattr(s, 'panel', (0, 0, 0, 0)) for s in step.products]

    scheme = ReactionScheme(steps)
    assert scheme._start == [steps1[0].reactants]
    assert scheme._end == [step.products for step in branched_steps]


def test_to_json_single_path():

    steps = [ReactionStep([r], [p], c) for r, p, c in zip(species[:3], species[1:4], conditions)][:2]
    true_json =  '[{"contents": [{"smiles": "a", "label": 0}], "successors": ' \
                 '[{"contents": "{0: 0}", "successors": ' \
                 '[{"contents": [{"smiles": "b", "label": 1}], "successors": ' \
                 '[{"contents": "{1: 1}", "successors": ' \
                 '[{"contents": [{"smiles": "c", "label": 2}], "successors": ' \
                 'null}]}]}]}]}]'

    scheme = ReactionScheme(steps)
    scheme._start = [steps[0].reactants]

    assert scheme.to_json() == true_json


def test_to_json_separate_paths():
    steps1 = [ReactionStep([r], [p], c) for r, p, c in zip(species[:3], species[1:4], conditions)][:2]
    steps2 = [ReactionStep([r], [p], c) for r, p, c in zip(species[3:6], species[4:7], conditions[2:4])][:2]
    steps = steps1 + steps2
    scheme = ReactionScheme(steps)
    scheme._start = [steps1[0].reactants, steps2[0].reactants]
    true_json = '[{"contents": [{"smiles": "a", "label": 0}], "successors": ' \
                '[{"contents": "{0: 0}", "successors": ' \
                '[{"contents": [{"smiles": "b", "label": 1}], "successors": ' \
                '[{"contents": "{1: 1}", "successors": ' \
                '[{"contents": [{"smiles": "c", "label": 2}], "successors": null}]}]}]}]}, ' \
                '{"contents": [{"smiles": "d", "label": 3}], "successors": ' \
                '[{"contents": "{2: 2}", "successors": ' \
                '[{"contents": [{"smiles": "e", "label": 4}], "successors": ' \
                '[{"contents": "{3: 3}", "successors": ' \
                '[{"contents": [{"smiles": "f", "label": 5}], "successors": null}]}]}]}]}]'
    assert  scheme.to_json() == true_json


def test_to_json_multiple_products():
    steps1 = [ReactionStep([r], [p], c) for r, p, c in zip(species[:3], species[1:4], conditions)][:2]
    branch_group = steps1[-1].products
    branched_steps = [ReactionStep(branch_group, [p], c) for p, c in zip(species[3:6], conditions[2:4])][:3]
    steps = steps1+branched_steps
    scheme = ReactionScheme(steps)
    scheme._start = [steps1[0].reactants]
    true_json = '[{"contents": [{"smiles": "a", "label": 0}], "successors": ' \
                '[{"contents": "{0: 0}", "successors": ' \
                '[{"contents": [{"smiles": "b", "label": 1}], "successors": ' \
                '[{"contents": "{1: 1}", "successors": ' \
                '[{"contents": [{"smiles": "c", "label": 2}], "successors": ' \
                '[{"contents": "{2: 2}", "successors": [{"contents": [{"smiles": "d", "label": 3}], "successors": null}]}, ' \
                '{"contents": "{3: 3}", "successors": [{"contents": [{"smiles": "e", "label": 4}], "successors": null}]}]}]}]}]}]}]'

    assert scheme.to_json() == true_json






if __name__ == '__main__':
    test_create_graph_single_path()
    test_create_graph_multiple_paths()
    test_set_start_end_nodes_single_path()
    test_set_start_end_nodes_separate_paths()
    test_set_start_end_nodes_multiple_products()
    test_to_json_single_path()
    test_to_json_separate_paths()
    test_to_json_multiple_products()

