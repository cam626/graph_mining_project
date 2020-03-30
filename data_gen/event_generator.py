import argparse
import networkx as nx
import random


def generateObservation(graph):
    return [random.choice(list(graph.nodes))]


def generateCommunication(graph):
    e = random.choice(list(graph.edges))

    return [e[0], e[1]]


def generateEventList(graph_filename, num_events, percentage_observation):
    graph = nx.read_edgelist(graph_filename)
    
    events = []
    for _ in range(num_events):
        event_type = ("observation" if random.random() < percentage_observation else "communication")
        temp = [event_type]

        remainder = (generateObservation(graph) if event_type == "observation" else generateCommunication(graph))
        temp.extend(remainder)
        events.append(temp)

    return events


def writeEventList(output_filename, events):
    with open(output_filename, "w+") as f:
        for event in events:
            f.write(" ".join(event) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_events", 
        help="The number of events to generate.",
        type=int,
        required=True)

    parser.add_argument("-g", "--graph_filename",
        help="The name of a file holding a graph edgelist.",
        required=True)

    parser.add_argument("-o", "--output_filename",
        help="The name of the file to output the event list to.",
        required=True)

    parser.add_argument("-p", "--percent_observation",
        help="The percentage of the events that should be observations of real data. \
            All remaining events will be communication between vertices.",
        type=float,
        required=True)

    args = parser.parse_args()
    events = generateEventList(args.graph_filename, args.num_events, args.percent_observation)
    writeEventList(args.output_filename, events)