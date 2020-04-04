import argparse
import json
import networkx as nx
import random

def normalized(func):
    def f(graph):
        centrality = func(graph)
        norm = max(centrality.values())
        for i in centrality:
            centrality[i] /= norm
        return centrality
    return f

class GraphManager():
    methods = {
        'degree': normalized(nx.degree_centrality),
        'closeness': normalized(nx.closeness_centrality),
        'betweenness': normalized(nx.betweenness_centrality),
        'eigenvector': normalized(nx.eigenvector_centrality)
        'random': specialCentrality(True)
        'sgd': specialCentrality(False)
    }


    def __init__(self, parameters):
        self.parameters = parameters

        self.graph = nx.read_edgelist(parameters["graph_filepath"])
        self.initializeData()

  
    def specialCentrality(self, rand):
        def f(graph):
            centrality = {}
            for n in graph:
                centrality[n] = random.uniform(0,1) if rand else 0.5
            return centrality
        return f


    def initializeData(self, post_train = False):
        self.beliefs = {}
        self.reliabilities = {}
        self.perceived = {}

        centrality = {}

        #initialize beliefs
        for n in self.graph:
            self.beliefs[n] = random.uniform(0.25,1)

        if (post_train):
            return

        #initialize vertex reliability
        centrality = methods[self.parameters.vertex_reliability_method](self.graph)
        for n in self.beliefs:
            self.reliabilities[n] = centrality[n]

        #initialize edge reliability
        if (self.parameters.vertex_reliability_method
            != self.parameters.edge_reliability_method):
            centrality = methods[self.parameters.edge_reliability_method](self.graph)
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
        if (self.parameters.edge_reliability_method == 'sgd'):
            self.graph.initializeData(True)
        events = self.graph.eventGenerator()
        for event in events:
            self.graph.processEvent(event)
        return self.graph.accuracy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", 
        help="The configuration file to use for the simulation.",
        required=True)

    args = parser.parse_args()

    sim = Simulator(args.config)
    sim.run()