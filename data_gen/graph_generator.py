import argparse
import networkx.generators.random_graphs as generators
import networkx as nx
import matplotlib.pyplot as plt


def barabasi_albert_generator(args):
    return generators.barabasi_albert_graph(args.num_nodes, args.num_edges)


def erdos_renyi_generator(args):
    return generators.erdos_renyi_graph(args.num_nodes, args.probability)


def watts_strogatz_generator(args):
    return generators.watts_strogatz_graph(args.num_nodes, args.k_nearest_neighbors, args.probability)


GRAPH_GENERATORS = {
    "barabasi_albert": barabasi_albert_generator,
    "erdos_renyi": erdos_renyi_generator,
    "watts_strogatz": watts_strogatz_generator
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_nodes", 
        help="The number of nodes in the generated graph.", 
        type=int,
        required=True)

    subparsers = parser.add_subparsers(help="Choose which type of graph to generate.", dest="type")

    barabasi_parser = subparsers.add_parser("barabasi_albert", help="Generate a Barabasi-Albert graph.")
    erdos_renyi = subparsers.add_parser("erdos_renyi", help="Generate a Erdos-Renyi graph.")
    watts_strogatz = subparsers.add_parser("watts_strogatz", help="Generate a Watts-Strogatz graph.")

    barabasi_parser.add_argument("-e", "--num_edges", 
        help="The number of edges in the generated graph.", 
        type=int,
        required=True)

    erdos_renyi.add_argument("-p", "--probability", 
        help="The probability of edge creation.",
        type=float,
        required=True)

    watts_strogatz.add_argument("-p", "--probability",
        help="The probability fo rewiring each edge.",
        type=float,
        required=True)

    watts_strogatz.add_argument("-k", "--k_nearest_neighbors",
        help="Each node is connected to k nearest neighbors in ring topology",
        type=int,
        required=True)

    args = parser.parse_args()

    generation_func = GRAPH_GENERATORS[args.type]
    graph = generation_func(args)

    suffix = ""
    for arg in vars(args):
        suffix += "_{}_{}".format(arg, getattr(args, arg))
    figure_filename = "fig" + suffix + ".png"
    edgelist_filename = "graph" + suffix + ".txt"

    nx.draw_kamada_kawai(graph, with_labels=True)
    plt.savefig(figure_filename)

    nx.write_edgelist(graph, edgelist_filename, data=False)