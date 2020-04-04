import argparse
import json
import networkx as nx
import random


class GraphManager():
    def __init__(self, parameters):
        self.parameters = parameters

        self.graph = nx.read_edgelist(parameters["graph_filepath"])
        self.initializeData()

  
    def initializeData(self):
        methods = {
            'degree': nx.degree_centrality,
            'closeness': nx.closeness_centrality,
            'betweenness': nx.betweenness_centrality,
            'eigenvector': nx.eigenvector_centrality
        }
        centrality = {}

        self.beliefs = {}
        self.reliabilities = {}
        self.perceived = {}

        #initialize beliefs
        for n in self.graph:
            self.beliefs[n] = random.uniform(0.25,1)

        #initialize vertex reliability
        if (self.parameters.vertex_reliability_method != 'random'):
            centrality = methods[self.parameters.vertex_reliability_method](self.graph)
        norm = max(centrality.values()) if len(centrality > 0) else 0
        for n in self.beliefs:
            if (self.parameters.vertex_reliability_method == 'random'):
                self.reliabilities[n] = random.uniform(0,1)
            else:
                self.reliabilities[n] = centrality[n] / norm

        #initialize edge reliability
        if (self.parameters.edge_reliability_method != 'sgd'):
            centrality = methods[self.parameters.edge_reliability_method](self.graph)
        norm = max(centrality.values()) if len(centrality > 0) else 0
        for e in self.graph.edges:
            if (e[0] not in self.perceived):
                self.perceived[e[0]] = {}
            if (e[1] not in self.perceived):
                self.perceived[e[1]] = {}
            if (self.parameters.edge_reliability_method == 'sgd'):
                self.perceived[e[0]][e[1]] = 0.5
                self.perceived[e[1]][e[0]] = 0.5
            else:
                self.perceived[e[0]][e[1]] = centrality[e[1]] / norm
                self.perceived[e[1]][e[0]] = centrality[e[0]] / norm   


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