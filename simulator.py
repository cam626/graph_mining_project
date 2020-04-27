'''
    Trust Based Information Diffusion Simulator

    Cameron Scott, Noah Prisament

    This file contains a simulator for a trust based information diffusion project
    that attempts to highlight flaws in the trust networks of society.

    April, 2020.
'''

import argparse
import json
import multiprocessing as mp
import random
import statistics

import networkx as nx
import numpy as np
from tqdm import tqdm

'''Specify how many processes to run the simulations on'''
POOL_SIZE = 11

NUM_EVENTS = 1000                   # How many events to run diffusion for
BATCH_SIZE = 100                    # The number of events to train on before evaluating convergence
LEARNING_RATE = 1e-3                # The rate at which SGD updates weights
LEARNING_RATE_UPDATE_TIME = 1e3     # The number of batches to train before decreasing the learning rate
EPSILON = 1e-3                      # The convergence threshold for SGD


class GraphManager():
    '''A controller for all graph operations'''

    def __init__(self, parameters):
        '''Set the graph up with all initial data and pruning.
        
        This includes parsing the graph file, pruning hubs of the graph
        and initializing vertex beliefs, vertex reliabilities and trust
        values on edges.
        '''
        self.parameters = parameters

        self.centrality_methods = {
            'degree': self.degreeCentrality,
            'closeness': nx.closeness_centrality,
            'betweenness': nx.betweenness_centrality,
            'eigenvector': self.eigenvectorCentrality,
            'random': self.randomCentrality,
            'sgd': self.sgdCentrality
        }

        G = nx.read_edgelist(parameters["graph_filepath"], create_using=nx.OrderedGraph())
        
        self.graph = nx.OrderedMultiDiGraph(G.to_directed())
        self.pruneGraph()
        
        self.initializeData()


    def pruneGraph(self):
        '''Randomly remove 80% of incoming edges from vertices defined
        as hubs.
        '''
        hubs = self.getHubs()
        for h in hubs:
            pre = list(self.graph.predecessors(h))
            reduced = len(pre) * 0.2
            while (self.graph.in_degree(h) > reduced):
                remove = random.choice(pre)
                pre.remove(remove)
                self.graph.remove_edge(remove, h)


    def getHubs(self):
        '''Identify and return all of the vertices that are classified as hubs
        in the graph.
        '''
        centrality = self.degreeCentrality(self.graph)
        cutoff = (sum(centrality.values()) / len(centrality)) + 1.5 * statistics.stdev(centrality.values())

        hubs = []
        for n in centrality:
            if (centrality[n] >= cutoff):
                hubs.append(n)
        return hubs


    def normalize(self, centrality):
        '''Normalize a dictionary of centrality values.
        '''
        norm = max(centrality.values())
        for i in centrality:
            centrality[i] /= norm
        return centrality


    def centrality(self, method):
        '''Apply a given centrality function to the graph and return the
        normalized results.
        '''
        func = self.centrality_methods[method]

        if method in ["sgd", "random", 'degree']:
            return func(self.graph)
        return self.normalize(func(self.graph))


    def degreeCentrality(self, graph):
        '''Compute the degree centrality of the graph.
        '''
        centrality = {}
        centrality_in = nx.in_degree_centrality(graph)
        centrality_out = nx.out_degree_centrality(graph)
        for n in centrality_in:
            centrality[n] = (centrality_in[n] + centrality_out[n]) / 2
        return centrality


    def eigenvectorCentrality(self, graph):
        '''Compute the eigenvector centrality of the graph.
        '''
        centrality = {}
        
        centrality_left = nx.eigenvector_centrality(graph)
        centrality_right = nx.eigenvector_centrality(graph.reverse())
        for n in centrality_left:
            centrality[n] = (centrality_left[n] + centrality_right[n]) / 2
        return centrality


    def randomCentrality(self, graph):
        '''Assign each vertex a random centrality value.
        '''
        centrality = {}
        for n in graph:
            centrality[n] = random.uniform(0,1)
        return centrality


    def sgdCentrality(self, graph):
        '''Assign each vertex a centrality value of 0.5.
        '''
        centrality = {}
        for n in graph:
            centrality[n] = 0.5
        return centrality


    def initializeBeliefs(self):
        '''Initialize the belief of each vertex randomly between 0.25 and 1.
        '''
        self.beliefs = {}

        for n in self.graph:
            self.beliefs[n] = random.uniform(0.25,1)


    def initializeData(self):
        '''Handle all data intialization on the graph including
        beliefs, vertex reliabilities and trust values.
        '''
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
        '''Create a single event randomly that send the belief of the sender
        to the receiver based on the senders reliability.'''
        edge = random.choice(list(self.graph.edges))
        
        data = self.beliefs[edge[0]]

        # The data that the sender believes may or may not be sent depending
        # on reliability
        data = (data if random.random() < self.reliabilities[edge[0]] else 1 - data)

        return (edge[0], edge[1], data)


    def eventGenerator(self, n):
        '''A generator that yields events until the number of events generated
        is equal to the number requested in the argument.'''
        for _ in range(n):
            yield self.event()


    def getPerceivedState(self):
        '''Return the edge states for the entire graph, in the same order each call.
        '''
        state = np.array([])
        for e in self.graph.edges:
            state = np.concatenate((state, np.array([self.perceived[e[0]][e[1]], self.perceived[e[1]][e[0]]])), axis=0)
        return state


    def error(self):
        '''Returns the Sum of Squared Errors of the current node beliefs.'''
        SSE = 0
        for i in self.beliefs:
            SSE += (1 - self.beliefs[i]) ** 2
        return SSE


    def processEvent(self, event):
        '''Update the belief of a receiving node based on the data received
        from the sender and the perceived reliability of the sender from
        the receivers perspective.
        '''
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
        '''Update the perceived reliability of the sender from the receiver's
        perspective based on SGD where the objective function is the squared
        error of the data received.

        w = w - eta * (-2 * (1 - wx) * x)
        '''
        sender = event[0]
        receiver = event[1]
        data = event[2]

        eta = LEARNING_RATE
        self.perceived[receiver][sender] = self.perceived[receiver][sender] - eta * \
            (-2 * (1 - self.perceived[receiver][sender] * data) * data)


    def accuracy(self, vertices=None):
        '''Returns the percentage of vertices with the correct value.'''
        count = 0

        if vertices == None:
            vertices = self.beliefs.keys()

        for i in vertices:
            if (self.beliefs[i] >= 0.5):
                count += 1

        return count / len(vertices)


    def hubNeighborhoodAccuracy(self):
        '''Returns the percentage of vertices in the neighborhood of a hub
        that have a truthful belief.
        '''
        hubs = self.getHubs()

        hub_neighbors = set()
        for hub in hubs:
            hub_neighbors = hub_neighbors.union(set(self.graph.neighbors(hub)))

        return self.accuracy(hub_neighbors)


class Simulator():
    '''A container for a single simulation.
    '''

    def __init__(self, config):
        self.parameters = config

        self.graph = GraphManager(self.parameters)


    def train(self):
        '''Continuously generate event batches until the perceived state of the 
        network is changing less than the convergence threshold.
        '''
        global LEARNING_RATE

        old_perceived_state = np.zeros(shape=(2*len(self.graph.graph.edges)))
        new_perceived_state = self.graph.getPerceivedState()

        batch_num = 1
        while np.linalg.norm(old_perceived_state-new_perceived_state) > EPSILON:
            old_perceived_state = new_perceived_state
            events = self.graph.eventGenerator(BATCH_SIZE)
            for event in events:
                self.graph.processEvent(event)
                self.graph.train(event)
            new_perceived_state = self.graph.getPerceivedState()

            # Step Decay Adaptive Learning Rate
            batch_num += 1
            if batch_num % LEARNING_RATE_UPDATE_TIME == 0:
                LEARNING_RATE /= 2


    def run(self):
        '''Train the network if necessary and then simulate the number of events
        needed.
        '''
        if (self.parameters["edge_reliability_method"] == 'sgd'):
            self.train()
            self.graph.initializeBeliefs()

        events = self.graph.eventGenerator(NUM_EVENTS)
        for event in events:
            self.graph.processEvent(event)

        return self.graph.accuracy()


class HyperSimulator():
    '''A container that runs simulations for all of the configurations
    in the configuration file a specified number of times and averages
    the results together for each configuration.
    '''

    def __init__(self, configs_filename, num_runs, output_file):
        self.num_runs = num_runs
        self.output_file = output_file

        with open(configs_filename, "r") as f:
            self.configurations = json.load(f)["configurations"]


    def runSimulation(self, config):
        '''Instantiate and run a single simulation with a given
        configuration.
        '''
        simulator = Simulator(config)
        accuracy = simulator.run()
        hub_neighborhood_accuracy = simulator.graph.hubNeighborhoodAccuracy()
        return {
            "accuracy": accuracy,
            "hub_neighborhood_accuracy": hub_neighborhood_accuracy
        }


    def nextSimulator(self):
        '''Generate the configurations that should be used for each
        simulation.
        '''
        for config_index, config in enumerate(self.configurations):
            for i in range(self.num_runs):
                print("Starting run #{} of configuration {}".format(i, config_index))
                yield config


    def run(self):
        '''Run all of the configurations the specified number of times in parallel.

        The number of processes to use is specified by POOL_SIZE. Output is saved to
        the given output file.
        '''
        with mp.Pool(POOL_SIZE) as p:
            results = list(p.map(self.runSimulation, self.nextSimulator()))

        output = {
            "results": []
        }

        for config_index, config in enumerate(self.configurations):
            partial_results = results[config_index * self.num_runs:(config_index + 1) * self.num_runs]

            temp_result = {
                "config": config
            }

            accuracies = []
            hub_neighborhood_accuracies = []

            for result in partial_results:
                accuracies.append(result["accuracy"])
                hub_neighborhood_accuracies.append(result["hub_neighborhood_accuracy"])

            temp_result["accuracies"] = accuracies
            temp_result["hub_neighborhood_accuracies"] = hub_neighborhood_accuracies

            temp_result["average_accuracy"] = sum(temp_result["accuracies"]) / self.num_runs
            temp_result["average_hub_neighborhood_accuracy"] = sum(temp_result["hub_neighborhood_accuracies"]) / self.num_runs

            if self.num_runs >= 2:
                temp_result["accuracy_standard_deviation"] = statistics.stdev(temp_result["accuracies"])
                temp_result["hub_neighborhood_accuracy_standard_deviation"] = statistics.stdev(temp_result["hub_neighborhood_accuracies"])
            else:
                temp_result["accuracy_standard_deviation"] = 0
                temp_result["hub_neighborhood_accuracy_standard_deviation"] = 0

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
