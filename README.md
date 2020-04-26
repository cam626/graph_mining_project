# Trust Based Information Diffusion

In the world that we live in, information is constantly flowing from sources to the rest of the world. People accept information based on how much they trust the source that it is coming from. In this simulation, we assign trust values between vertices based on different metrics such as vertex centrality as well as through stochastic gradient descent. These various methods of trust can help indicate the flaws of human decision making and can also give insight into how we should trust others.

## Initialization

To start, each vertex is given an initial state of what they believe is true. These beliefs are randomly initialized between 0.25 and 1, so that the network has a slight trend toward truth. Each vertex must also be given a reliability value that indicates how likely they are to communicate what they believe to be true when sending information to another vertex. These vertex reliability values can be based on the following properties:

- Degree Centrality
- Closeness Centrality
- Betweenness Centrality
- Random

Each edge between two communicating vertices must be initialized with a trust value that indicates how much the receiving vertex trusts the sending vertex. This trust value indicates how likely the receiving vertex is to accept information from the sending vertex. These trust values can be based on the following properties:

- Degree Centrality
- Closeness Centrality
- Betweenness Centrality
- Random
- Stochastic Gradient Descent (SGD)

If the trust values are based on SGD, there is a training period before the diffusive process that allows trust values to be learned.

## Diffusion

The information that is diffusing through the networks in this simulation is purely real numbers in the range from 0 to 1, where 1 indicates true information and 0 indicates false information. When a vertex receives information from another, it does not know whether it is true or false. It will accept the information based on how much it trusts the sending vertex and how stubborn it is. The higher the stubbornness value for the network, the less likely a vertex is to accept new information.

If the edge reliability method is set to SGD, the vertex will then 'fact check' the information given to it in order to update the trust value for the sending vertex. The update equation, derived from SGD, is shown below.

w<sub>t</sub> = w<sub>t-1</sub> - η * (-2 * (1 - w<sub>t-1</sub>x) * x)

Here `w` is the trust value, `η` is the learning rate, and `x` is the information received.

## Graphs

Two different graphs are used in the simulation, both of which were procedurally generated. One of the graphs is an Erdős–Rényi graph and the other is a Barabási–Albert graph. The Barabási–Albert graph is meant to simulate a real-world situation while the Erdős–Rényi graph was used as a control.

Hub pruning was applied to both of these graphs, which is where incoming edges are randomly removed from vertices identified as hubs. Hubs are defined as any vertices with a degree centrality greater that 1.5 standard deviations over the average degree centrality. 