import argparse
import json
import random
import networkx as nx


class GraphManager():
    def __init__(self, parameters):
        self.parameters = parameters

        self.centrality_methods = {
            'degree': self.normalized(nx.degree_centrality),
            'closeness': self.normalized(nx.closeness_centrality),
            'betweenness': self.normalized(nx.betweenness_centrality),
            'eigenvector': self.normalized(nx.eigenvector_centrality),
            'random': self.specialCentrality(method="random"),
            'sgd': self.specialCentrality()
        }

        self.graph = nx.read_edgelist(parameters["graph_filepath"])
        self.initializeData()


    def normalized(self, func):
        def f(graph):
            centrality = func(graph)
            norm = max(centrality.values())
            for i in centrality:
                centrality[i] /= norm
            return centrality
        return f

  
    def specialCentrality(self, method="sgd"):
        def f(graph):
            centrality = {}
            for n in graph:
                centrality[n] = random.uniform(0,1) if method == "random" else 0.5
            return centrality
        return f


    def initializeBeliefs(self):
        self.beliefs = {}

        for n in self.graph:
            self.beliefs[n] = random.uniform(0.25,1)


    def initializeData(self):
        self.reliabilities = {}
        self.perceived = {}

        # Each vertex needs to start with a belief of what is true
        self.initializeBeliefs()

        # Initialize vertex reliability based on the method given
        centrality = self.centrality_methods[self.parameters["vertex_reliability_method"]](self.graph)
        for n in self.graph:
            self.reliabilities[n] = centrality[n]

        # Initialize edge reliability based on the method given
        # Only recalculate centrality if the method is different than that of vertex reliability
        if (self.parameters["vertex_reliability_method"] != self.parameters["edge_reliability_method"]):
            centrality = self.centrality_methods[self.parameters["edge_reliability_method"]](self.graph)

        for e in self.graph.edges:
            if (e[0] not in self.perceived):
                self.perceived[e[0]] = {}
            if (e[1] not in self.perceived):
                self.perceived[e[1]] = {}

            self.perceived[e[0]][e[1]] = centrality[e[1]]
            self.perceived[e[1]][e[0]] = centrality[e[0]]


    def event(self):
        edge = random.choice(list(self.graph.edges))
        
        return (edge[0], edge[1], self.beliefs[edge[0]])


    def eventGenerator(self):
        for _ in range(self.parameters["num_events"]):
            yield self.event()


    def error(self):
        """returns the Sum of Squared Errors"""
        SSE = 0
        for i in self.beliefs:
            SSE += (1 - self.beliefs[i]) ** 2
        return SSE


    def processEvent(self, event):
        pass


    def train(self, event):
        pass


    def accuracy(self):
        """returns the percentage of vertices
           with the correct value"""
        sum = 0
        for i in self.beliefs:
            if (self.beliefs[i] >= 0.5):
                sum += 1
        return sum / len(self.beliefs)


class Simulator():
    def __init__(self, config_filename):
        self.parseConfig(config_filename)

        self.graph = GraphManager(self.parameters)


    def parseConfig(self, config_filename):
        with open(config_filename, "r") as f:
            self.parameters = json.load(f)


    def train(self):
        while self.graph.error() > self.parameters["epsilon"]:
            event = self.graph.event()
            self.graph.processEvent(event)
            self.graph.train(event)


    def run(self):
        if (self.parameters["edge_reliability_method"] == 'sgd'):
            self.train()
            self.graph.initializeBeliefs()

        events = self.graph.eventGenerator()
        for event in events:
            self.graph.processEvent(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", 
        help="The configuration file to use for the simulation.",
        required=True)

    args = parser.parse_args()

    sim = Simulator(args.config)
    sim.run()

    print("Simulation ended with an accuracy of {}.".format(sim.graph.accuracy()))
