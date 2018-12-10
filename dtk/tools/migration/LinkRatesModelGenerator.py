import abc
from dtk.tools.migration.GraphGenerator import GraphGenerator


class LinkRatesModelGenerator:
    """
    A base abstract LinkRatesModelGenerator. This is used to generate the link rates for a model

    """

    @abc.abstractmethod
    def generate(self) -> dict:
        """
        Generates the link rates and returns as a dictionary that represents a weighted adjacency list

        Returns:
            weighted adjacency list
        """
        pass


class GraphGeneratedLinkRatesModelGenerator(LinkRatesModelGenerator):
    """
    Abastract class that represents LinkRatesModelGenerators that required a GraphGenerator the adjacency list
    """
    def __init__(self, graph_generator: GraphGenerator):
        self.graph_generator = graph_generator
        self.graph_topo = None
        self.link_rates = None

        # graph topology instance
        self.gt = None

    @abc.abstractmethod
    def generate(self) -> dict:
        """"
        Generates the link rates and returns as a dictionary that represents a weighted adjacency list

        Returns:
            weighted adjacency list
        """
        pass
