"""
This file contains data structures required to represent the output of ReactionDataExtractor
"""

from abc import ABC, abstractmethod
from collections import Counter
import json
from models.reaction import Reactant, Product


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

        self._single_path = True if all(len(graph[item]) == 1 or item == self.products for item in graph) else False
        # Simpler __str__ if an image consists of a single reaction path

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
                       all(isinstance(species, Reactant) for species in group)]

        self._end = [group for group, count in group_count.items() if count == 1 and
                     all(isinstance(species, Product) for species in group)]

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








