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
        self.beliefs = {}
        self.perceived = {}
        self.reliabilities = {}


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
        count = 0
        for i in self.beliefs:
            if (self.beliefs[i] >= 0.5):
                count += 1
        return count / len(self.beliefs)


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


class HyperSimulator():
    def __init__(self, config_filename):
        self.n = 100
        self.simulators = []
        for _ in range(self.n):
            self.simulators.append(Simulator(config_filename))


    def run(self):
        sum = 0
        for sim in self.simulators:
            if (sim.parameters.edge_reliability_method == 'sgd'):
                sim.train()
            sum += sim.run()
        print(sum / self.n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", 
        help="The configuration file to use for the simulation.",
        required=True)

    args = parser.parse_args()

    sim = Simulator(args.config)
    sim.run()