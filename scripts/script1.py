import json
import matplotlib.pyplot as plt

with open("../output.json", "r") as f:
    data = json.load(f)["results"]

stub = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

plt.ylabel("Average Accuracy")
plt.xlabel("Vertex Stubborness")

def makeGraphs(list, graph):
    # vertex centralities

    centr = ['degree', 'closeness', 'betweenness']
    
    for j in centr:
        g = [i for i in list if i['config']["vertex_reliability_method"] == j]
    
        edge = [j, "sgd"]
    
        for e in edge:
            plt.plot(stub, [i["average_accuracy"] for i in g if i['config']["edge_reliability_method"] == e], label=e) 
        
        plt.ylabel("Average Accuracy")
        plt.xlabel("Vertex Stubborness")
        plt.legend(loc='upper left')   
        if (j == 'degree'):
            plt.title("Degree Vertex Reliability" + " " + graph)
        if (j == 'closeness'):
            plt.title("Closeness Vertex Reliability" + " " + graph)
        if (j == 'betweenness'):
            plt.title("Betweenness Vertex Reliability" + " " + graph)
        plt.show() 

    # vertex random

    g = [i for i in list if i['config']["vertex_reliability_method"] == 'random']

    edge = ['random', 'degree', 'closeness', 'betweenness', 'sgd']

    for e in edge:
        plt.plot(stub, [i["average_accuracy"] for i in g if i['config']["edge_reliability_method"] == e], label=e) 

        plt.ylabel("Average Accuracy")
        plt.xlabel("Vertex Stubborness")
        plt.legend(loc='upper left')       
        plt.title("Random Vertex Reliability" + " " + graph)
    
    plt.show()    

### barabasi-albert ###

ba = [i for i in data if i['config']["graph_filepath"] == "data/barabasi_albert/graph_num_nodes_100_type_barabasi_albert_num_edges_4.txt"]

makeGraphs(ba, "Barabasi-Albert")

### erdos-renyi ###

er = [i for i in data if i['config']["graph_filepath"] == "data/erdos_renyi/graph_num_nodes_100_type_erdos_renyi_probability_0.2.txt"]

makeGraphs(er, "Erdos-Renyi")
