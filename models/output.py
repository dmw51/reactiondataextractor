"""
This file contains data structures required to represent the output of ReactionDataExtractor
"""

from abc import ABC, abstractmethod
from collections import Counter
import json
from models.reaction import ChemicalStructure, ChemicalStructure


from models.reaction import Conditions


class Graph(ABC):
    """
    Generic directed graph class
    """

    def __init__(self, graph_dict=None):
        if graph_dict is None:
            graph_dict = {}
        self._graph_dict = graph_dict

    @abstractmethod
    def _generate_edge(self, *args) :
        """
        This method should update the graph dict with a connection between vertices,
        possibly adding some edge annotation.
        """
        return NotImplemented

    @abstractmethod
    def edges(self):
        """
        This method should return all edges (partially via invoking the `_generate_edge` method).
        """
        return NotImplemented

    @abstractmethod
    def __str__(self):
        """
        A graph needs to have a __str__ method to constitute a valid output representation.
        """

    def vertices(self):
        return self._graph_dict.keys()

    def add_vertex(self, vertex):
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def find_isolated_vertices(self):
        """
        Returns all isolated vertices. Can be used for output validation
        :return: collection of isolated (unconnected) vertices
        """
        graph = self._graph_dict
        return [key for key in graph if graph[key] == []]

    def find_path(self, vertex1, vertex2, path=None):
        if path is None:
            path = []
        path += [vertex1]
        graph = self._graph_dict
        if vertex1 not in graph:
            return None
        if vertex2 in graph[vertex1]:
            return path + [vertex2]
        else:
            for value in graph[vertex1]:
                return self.find_path(value, vertex2, path)


class ReactionScheme(Graph):
    def __init__(self, reaction_steps):
        self._reaction_steps = reaction_steps
        super().__init__()
        self.create_graph()
        self._start = None  # start node(s) in a graph
        self._end = None   # end node(s) in a graph
        graph = self._graph_dict
        self.set_start_end_nodes()

        # self._single_path = True if all(len(graph[item]) == 1 or self.products == item for item in graph) else False
        self._single_path = True if len(self._start) == 1 and len(self._end) == 1 else False

    def edges(self):
        if not self._graph_dict:
            self.create_graph()

        return {k: v for k, v in self._graph_dict.items()}

    def _generate_edge(self, key, successor):

        self._graph_dict[key].append(successor)

    def __repr__(self):
        return f'ReactionScheme({self._reaction_steps})'

    def __str__(self):
        # if self._single_path:
        #     path = self.find_path(self.reactants, self.products)
        #     return '  --->  '.join((' + '.join(str(species) for species in group)) for group in path)
        # else:
        return str(self._graph_dict)

    def __eq__(self, other):
        if isinstance(other, ReactionScheme):
            return other._graph_dict == self._graph_dict
        return False

    @property
    def graph(self):
        # if not self._graph_dict:
        #     self.create_graph()

        return self._graph_dict

    @property
    def reactants(self):
        return self._start

    @property
    def products(self):
        return self._end

    def create_graph(self):
        """
        Unpack reaction steps to create a graph from individual steps
        :return: completed graph dictionary
        """
        graph = self._graph_dict
        for step in self._reaction_steps:
            [self.add_vertex(frozenset(species_group)) for species_group in step]
            self.add_vertex(step.conditions)

        for step in self._reaction_steps:
            self._generate_edge(step.reactants, step.conditions)
            self._generate_edge(step.conditions, step.products)

        return graph

    def set_start_end_nodes(self):
        """
        Finds and return the first vertex in a graph (group of reactants). Unpack all groups from ReactionSteps into
        a Counter. The first group is a group that is counted only once and exists as a key in the graph dictionary.
        Other groups (apart from the ultimate products) are counted twice (as a reactant in one step and a product in
        another).
        """
        group_count = Counter(group for step in self._reaction_steps for group in (step.reactants, step.products))
        self._start = [group for group, count in group_count.items() if count == 1 and
                       all(isinstance(species, ChemicalStructure) for species in group)]

        self._end = [group for group, count in group_count.items() if count == 1 and
                     all(isinstance(species, ChemicalStructure) for species in group)]

    def find_path(self, group1, group2, path=None):
        graph = self._graph_dict
        if path is None:
            path = []
        path += [group1]
        if group1 not in graph:
            return None

        successors = graph[group1]
        if group2 in successors:
            return path+[group2]
        else:
            for prod in successors:
                return self.find_path(prod, group2, path=path)
        return None

    def to_json(self):
        reactions = [self._json_generic_recursive(start_node) for start_node in self._start]

        return json.dumps(reactions)

    # def _json_single_path(self, key=None, json_obj=None):
    #     """
    #     Recursive method for constructing json objects by traversing a simple (single reaction path) graph
    #     :param key: key used to access a value in ``self._graph_dict``
    #     :return: constructed JSON object
    #     """
    #     graph = self._graph_dict
    #     if key is None:
    #         key = self._first
    #
    #     if json_obj is None:
    #         json_obj = {}
    #
    #     node = key
    #
    #     if hasattr(node, '__iter__'):
    #         contents = [{'smiles': species.smiles, 'label': species.label} for species in node]
    #     else:
    #         contents = str(node)   # Convert the conditions_dct directly
    #     json_obj['contents'] = contents
    #
    #     try:
    #         successor = graph[node][0]
    #         json_obj['successors'] = self._json_single_path(successor)
    #     except IndexError:
    #         json_obj['successors'] = None
    #         return json_obj
    #
    #     return json_obj

    def _json_generic_recursive(self, start_key, json_obj=None):
        """
        Generic recursive json string generator. Takes in a single ``start_key`` node and builds up the ``json_obj`` by
        traverding the reaction graph
        :param start_key: node where the traversal begins (usually the 'first' group of reactants in the reactions)
        :param json_obj: a dictionary created in the recursive procedure (ready for json dumps)
        :return:  dict; the created ``json_obj``
        """
        graph = self._graph_dict

        if json_obj is None:
            json_obj = {}

        node = start_key

        if hasattr(node, '__iter__'):
            contents = [{'smiles': species.smiles, 'label': species.label} for species in node]
        else:
            contents = str(node)   # Convert the conditions_dct directly

        json_obj['contents'] = contents
        successors = graph[node]
        if not successors:
            json_obj['successors'] = None
            return json_obj
        else:
            json_obj['successors'] = []
            for successor in successors:
                json_obj['successors'].append(self._json_generic_recursive(successor))

        return json_obj

    def to_smirks(self, start_key=None, species_strings=None):
        """
        Converts the reaction graph into a SMIRKS (or more appropriately - reaction SMILES, its subset). Also outputs
        a string containing auxiliary information from the conditions' dictionary.
        :param start_key: node where the traversal begins (usually the 'first' group of reactants in the reactions)
        :param species_strings: list of found smiles strings (or chemical formulae) built up in the procedure and ready
        for joining into a single SMIRKS string.
        :return: (str, str) tuple containing a (reaction smiles, auxiliary info) pair
        """
        if not self._single_path:
            return NotImplemented  # SMIRKS only work for single-path reaction

        graph = self._graph_dict

        if start_key is None:
            start_key = self._start[0]

        if species_strings is None:
            species_strings = []

        node = start_key

        if hasattr(node, '__iter__'):  # frozenset of reactants or products
            species_str = '.'.join(species.smiles for species in node)
        else:  # Conditions object
            # The string is a sum of coreactants, catalysts (which have small dictionaries holding names and values/units)
            species_vals = '.'.join(species_dct['Species'] for group in iter((node['coreactants'], node['catalysts'],
                                                               )) for species_dct in group)
            # and auxiliary species with simpler structures (no units)
            species_novals = '.'.join(group for group in node['other species'] )
            species_str = '.'.join(filter(None, [species_vals, species_novals]))

        species_strings.append(species_str)

        successors = graph[node]
        if not successors:
            smirks ='>'.join(species_strings)
            return smirks
        else:
            return self.to_smirks(successors[0], species_strings)

        return smirks, [node.conditions_dct for node in graph if isinstance(node, Conditions)]
