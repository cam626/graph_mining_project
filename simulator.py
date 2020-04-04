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
        """Create a single event randomly that send the belief of the sender
        to the receiver based on the senders reliability."""
        edge = random.choice(list(self.graph.edges))
        
        data = self.beliefs[edge[0]]

        # The data that the sender believes may or may not be sent depending
        # on reliability
        data = (data if random.random() < self.reliabilities[edge[0]] else 1 - data)

        return (edge[0], edge[1], data)


    def eventGenerator(self):
        """A generator that yields events until the number of events generated
        is equal to the number requested in the parameters."""
        for _ in range(self.parameters["num_events"]):
            yield self.event()


    def error(self):
        """Returns the Sum of Squared Errors of the current node beliefs."""
        SSE = 0
        for i in self.beliefs:
            SSE += (1 - self.beliefs[i]) ** 2
        return SSE


    def processEvent(self, event):
        """Update the belief of a receiving node based on the data received
        from the sender and the perceived reliability of the sender from
        the receivers perspective.
        """
        sender = event[0]
        receiver = event[1]
        data = event[2]

        # How the receiver perceives the senders trustworthiness
        perceived_reliability = self.perceived[receiver][sender]
        
        # Flip the data if the receiver chooses not to believe the sender
        data = (data if random.random() < perceived_reliability else 1 - data)

        # Update the receiver's belief as a weighted average of their current
        # belief and the received data
        vertex_stubbornness = self.parameters["vertex_stubbornness"]
        self.beliefs[receiver] = self.beliefs[receiver] * vertex_stubbornness + data * (1 - vertex_stubbornness)


    def train(self, event):
        pass


    def accuracy(self):
        """Returns the percentage of vertices with the correct value."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", 
        help="The configuration file to use for the simulation.",
        required=True)

    args = parser.parse_args()

    sim = Simulator(args.config)
    sim.run()