import multiprocessing as mp
import argparse
import json
import random
import statistics

import numpy as np
import networkx as nx
from tqdm import tqdm

POOL_SIZE = 11


class GraphManager():
    def __init__(self, parameters):
        self.parameters = parameters

        self.centrality_methods = {
            'degree': self.degreeCentrality,
            'closeness': nx.closeness_centrality,
            'betweenness': nx.betweenness_centrality,
            'eigenvector': self.eigenvectorCentrality,
            'random': self.randomCentrality,
            'sgd': self.sgdCentrality
        }

        G = nx.read_edgelist(parameters["graph_filepath"], create_using=nx.OrderedDiGraph())
        
        self.graph = nx.union(G, G.reverse())
        self.pruneGraph()
        
        self.initializeData()


    def pruneGraph(self):
        #for each hub, reduce it's in-degree, while loop and random choice
        # hubs = self.getHubs()
        # for h in hubs:
        #     while self.graph
        pass


    def getHubs(self):
        centrality = self.degreeCentrality(self.graph)
        cutoff = (sum(centrality.values()) / len(centrality)) + 1.5 * statistics.stdev(centrality.values())

        hubs = []
        for n in centrality:
            if (centrality[n] >= cutoff):
                hubs.append(n)
        return hubs


    def normalize(self, centrality):
        norm = max(centrality.values())
        for i in centrality:
            centrality[i] /= norm
        return centrality


    def centrality(self, method):
        func = self.centrality_methods[method]

        if method in ["sgd", "random", 'degree']:
            return func(self.graph)
        return self.normalize(func(self.graph))


    def degreeCentrality(self, graph):
        centrality = {}
        centrality_in = nx.in_degree_centrality(graph)
        centrality_out = nx.out_degree_centrality(graph)
        for n in centrality_in:
            centrality[n] = (centrality_in[n] + centrality_out[n]) / 2
        return centrality


    def eigenvectorCentrality(self, graph):
        centrality = {}
        centrality_left = nx.eigenvector_centrality(graph)
        centrality_right = nx.eigenvector_centrality(graph.reverse())
        for n in centrality_in:
            centrality[n] = (centrality_left[n] + centrality_right[n]) / 2
        return centrality


    def randomCentrality(self, graph):
        centrality = {}
        for n in graph:
            centrality[n] = random.uniform(0,1)
        return centrality


    def sgdCentrality(self, graph):
        centrality = {}
        for n in graph:
            centrality[n] = 0.5
        return centrality


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
        centrality = self.centrality(self.parameters["vertex_reliability_method"])
        for n in self.graph:
            self.reliabilities[n] = centrality[n]

        # Initialize edge reliability based on the method given
        # Only recalculate centrality if the method is different than that of vertex reliability
        if (self.parameters["vertex_reliability_method"] != self.parameters["edge_reliability_method"]):
            centrality = self.centrality(self.parameters["edge_reliability_method"])

        for e in self.graph.edges:
            if (e[0] not in self.perceived):
                self.perceived[e[0]] = {}
            if (e[1] not in self.perceived):
                self.perceived[e[1]] = {}

            self.perceived[e[0]][e[1]] = centrality[e[1]]
            self.perceived[e[1]][e[0]] = centrality[e[0]]


    def event(self):
        """Create a single event randomly that send the belief of the sender
        to the receiver based on the senders reliability."""
        edge = random.choice(list(self.graph.edges))
        
        data = self.beliefs[edge[0]]

        # The data that the sender believes may or may not be sent depending
        # on reliability
        data = (data if random.random() < self.reliabilities[edge[0]] else 1 - data)

        return (edge[0], edge[1], data)


    def eventGenerator(self, n):
        """A generator that yields events until the number of events generated
        is equal to the number requested in the argument."""
        for _ in range(n):
            yield self.event()


    def getPerceivedState(self):
        state = np.array([])
        for e in self.graph.edges:
            state = np.concatenate((state, np.array([self.perceived[e[0]][e[1]], self.perceived[e[1]][e[0]]])), axis=0)
        return state


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
        """Update the perceived reliability of the sender from the receiver's
        perspective based on SGD where the objective function is the squared
        error of the data received.

        w = w - eta * (-2 * (1 - wx) * x)
        """
        sender = event[0]
        receiver = event[1]
        data = event[2]

        eta = self.parameters["learning_rate"]
        self.perceived[receiver][sender] = self.perceived[receiver][sender] - eta * \
            (-2 * (1 - self.perceived[receiver][sender] * data) * data)


    def accuracy(self):
        """Returns the percentage of vertices with the correct value."""
        count = 0
        for i in self.beliefs:
            if (self.beliefs[i] >= 0.5):
                count += 1
        return count / len(self.beliefs)


class Simulator():
    def __init__(self, config):
        self.parameters = config

        self.graph = GraphManager(self.parameters)


    def train(self):
        old_perceived_state = np.zeros(shape=(2*len(self.graph.graph.edges)))
        new_perceived_state = self.graph.getPerceivedState()
        while np.linalg.norm(old_perceived_state-new_perceived_state) > self.parameters["epsilon"]:
            # print(np.linalg.norm(old_perceived_state-new_perceived_state))
            old_perceived_state = new_perceived_state
            events = self.graph.eventGenerator(self.parameters["batch_size"])
            for event in events:
                self.graph.processEvent(event)
                self.graph.train(event)
            new_perceived_state = self.graph.getPerceivedState()


    def run(self):
        if (self.parameters["edge_reliability_method"] == 'sgd'):
            self.train()
            self.graph.initializeBeliefs()

        events = self.graph.eventGenerator(self.parameters["num_events"])
        for event in events:
            self.graph.processEvent(event)

        return self.graph.accuracy()


class HyperSimulator():
    def __init__(self, configs_filename, num_runs, output_file):
        self.num_runs = num_runs
        self.output_file = output_file

        with open(configs_filename, "r") as f:
            self.configurations = json.load(f)["configurations"]


    def runSimulation(self, simulator):
        return simulator.run()


    def nextSimulator(self):
        for config_index, config in enumerate(self.configurations):
            for i in range(self.num_runs):
                print("Starting run #{} of configuration {}".format(i, config_index))
                yield Simulator(config)


    def run(self):
        with mp.Pool(POOL_SIZE) as p:
            results = p.map(self.runSimulation, self.nextSimulator())

        output = {
            "results": []
        }

        for config_index, config in enumerate(self.configurations):
            temp_result = {
                "config": config,
                "accuracies": results[config_index * self.num_runs:(config_index + 1) * self.num_runs]
            }

            temp_result["average"] = sum(temp_result["accuracies"]) / self.num_runs
            temp_result["standard_deviation"] = statistics.stdev(temp_result["accuracies"])

            output["results"].append(temp_result)

        with open(self.output_file, "w+") as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--configs", 
        help="A file containing the configurations to use.",
        required=True)

    parser.add_argument("-n", "--num_runs",
        help="The number of times to run each given configuration.",
        default=1,
        type=int)

    parser.add_argument("-o", "--output_file",
        help="The name of the file to output to.",
        required=True)

    args = parser.parse_args()

    sim = HyperSimulator(args.configs, args.num_runs, args.output_file)
    sim.run()
