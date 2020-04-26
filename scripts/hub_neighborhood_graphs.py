import json
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python [results_filename]")
        exit(1)

    filename = sys.argv[1]

    with open(filename, "r") as f:
        data = json.load(f)

    plot_data = {}

    for result in data["results"]:
        stub = result["config"]["vertex_stubbornness"]
        v_rel = result["config"]["vertex_reliability_method"]
        e_rel = result["config"]["edge_reliability_method"]

        graph_key = v_rel + "_" + e_rel
        graph_key_inner = result["config"]["graph_filepath"].split("/")[1]

        if graph_key not in plot_data:
            plot_data[graph_key] = {}

        if graph_key_inner not in plot_data[graph_key]:
            plot_data[graph_key][graph_key_inner] = {
                "accuracy": [0] * 11,
                "hub_accuracy": [0] * 11
            }

        index = int(stub * 10)

        accuracy = result["average_accuracy"]
        hub_accuracy = result["average_hub_neighborhood_accuracy"]
        plot_data[graph_key][graph_key_inner]["accuracy"][index] = accuracy
        plot_data[graph_key][graph_key_inner]["hub_accuracy"][index] = hub_accuracy

    x_axis = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for graph_type, graphs in plot_data.items():
        for graph_name, graph_data in graphs.items():
            title = (graph_name.replace("_", " ") + ": " + graph_type.replace("_", " ")).title()
            plt.clf()
            accuracy_line, = plt.plot(x_axis, graph_data["accuracy"], "r")
            hub_accuracy_line, = plt.plot(x_axis, graph_data["hub_accuracy"], "b")

            plt.legend([accuracy_line, hub_accuracy_line], ["Accuracy", "Hub Neighborhood Accuracy"])

            plt.ylabel("Accuracy (%)")
            plt.xlabel("Vertex Stubbornness (%)")
            plt.title(title)
            plt.savefig("plots/" + title)