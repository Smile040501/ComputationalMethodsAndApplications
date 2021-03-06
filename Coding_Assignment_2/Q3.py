"""
111901030
Mayank Singla
Coding Assignment 2 - Q3
"""

# %%
from sys import maxsize as INF
import matplotlib.pyplot as plt
from random import random
from math import log
from queue import Queue
from numpy import linspace


def handleError(method):
    """
    Decorator Factory function.
    Returns a decorator that normally calls the method of a class by forwarding all its arguments to the method.
    It surrounds the method calling in try-except block to handle errors gracefully.
    """

    def decorator(ref, *args, **kwargs):
        """
        Decorator function that surrounds the method of a class in try-except block and call the methods and handles error gracefully.
        """
        try:
            # Return the same value as that of the method if any
            return method(ref, *args, **kwargs)
        except Exception as err:
            print(type(err))
            print(err)

    return decorator


class UndirectedGraph:
    """
    Represents an Undirected Graph
    """

    @handleError
    def __init__(self, n=INF):
        """
        Constructor for the class Undirected Graph.\n
        Takes an optional argument n which is the number of nodes in the graph pre-defined.\n
        If n is not provided, the graph is Free.
        """
        if (not isinstance(n, int)) or (n < 0):
            # Checking for a valid
            raise Exception("The number of vertices must be a non-negative integer.")
        self.gr = {}  # Adjacency list as dictionary of (node, neighbours)
        self.maxNodes = n  # Maximum number of nodes graph could have
        self.numNodes = 0  # Current number of nodes in the graph
        self.numEdges = 0  # Current number of edges in the graph
        if n == INF:  # Returning if it is a Free graph
            return
        self.numNodes = n  # Current number of nodes in a non-free graph
        # Initializing the adjacency list for all the nodes in a non-free graph
        for i in range(1, n + 1):
            self.gr[i] = set()

    @handleError
    def _validateNode(self, node):
        """
        Validates the value of a given node.
        """
        # Node value should a positive integer
        if (not isinstance(node, int)) or (node <= 0):
            raise Exception("Node index must be a positive integer.")
        # Node value should be from {1 ... self.maxNodes}
        if node > self.maxNodes:
            raise Exception("Node index cannot exceed number of nodes")
        # Returning True on successful validation
        return True

    @handleError
    def addNode(self, node):
        """
        Adds a node to the graph, raising an exception if the graph is not free
        """
        # Validating the value of the given node
        if not self._validateNode(node):
            return
        # Adding the node to the graph if it is not already present in the graph
        if not node in self.gr:
            self.gr[node] = set()
            # Incrementing the count of the number of nodes in the graph
            self.numNodes += 1

    @handleError
    def _validateEdge(self, edge):
        """
        Validates an input edge for the graph.
        """
        if len(edge) != 2:  # Edge must contain 2 node values
            return False
        node1, node2 = edge  # Extracting the node value out of the edge
        # Validating the value of both the nodes of the edge
        if not self._validateNode(node1) or not self._validateNode(node2):
            return False
        # Returning True on successful validation
        return True

    @handleError
    def addEdge(self, a, b):
        """
        Adds an edge to the graph, and thereby also adding the nodes of the edges to the graph if not present
        """
        if not self._validateEdge((a, b)):  # Validating the edge
            return
        # Adding nodes of the edge to the graph
        self.addNode(a)
        self.addNode(b)
        # Adding both the nodes to each others' adjacency list as it is an undirected graph
        self.gr[a].add(b)
        self.gr[b].add(a)
        # Incrementing the number of edges in the graph
        self.numEdges += 1

    @handleError
    def __add__(self, t):
        """
        Overloading the + operator for the graph object so that it is possible to add nodes/edges to the graph
        """
        if isinstance(t, int):
            # If input is a single integer, add node to the graph
            self.addNode(t)
        elif isinstance(t, tuple) and len(t) == 2:
            # If input is a tuple of 2 nodes, add edge to the graph
            self.addEdge(t[0], t[1])
        else:
            # Any other input is invalid and operation is not supported
            raise Exception("Operation not supported.")
        return self

    @handleError
    def __str__(self):
        """
        Function that prints the object of this class in the expected format
        """
        desc = "Graph with {numNodes} nodes and {numEdges} edges. Neighbours of the nodes are belows:\n".format(
            numNodes=self.numNodes,
            numEdges=self.numEdges,
        )
        for key, val in self.gr.items():
            desc += "Node {node}: {neighbours}\n".format(
                node=key, neighbours="{}" if len(val) == 0 else str(val)
            )
        return desc

    @handleError
    def _calcDegreeDist(self):
        """
        Evaluates the degree distribution of the nodes of the graph.\n
        Returns: dictionary of (degree, number of nodes with that degree)
        """
        degreeDist = {}  # Initializing the dictionary
        for val in self.gr.values():  # Looping all the adjacency list of all the nodes
            degree = len(val)  # Length of the list is the degree of the node
            # Incrementing the count of this degree in the final dictionary
            if not degree in degreeDist:
                degreeDist[degree] = 0
            degreeDist[degree] += 1
        return degreeDist

    @handleError
    def plotDegDist(self):
        """
        Plots the degree distribution of the graph
        """
        # Giving title and labels to the graph
        plt.title("Node Degree Distribution")
        plt.xlabel("Node degree")
        plt.ylabel("Fraction of nodes")

        # Getting the degree distribution of the graph
        degreeDist = self._calcDegreeDist()

        # xpoints are degrees from {0 ... self.numNodes - 1}
        xpoints = [0] * self.numNodes
        # ypoints are the fraction of nodes in the graph with that degree
        ypoints = [0] * self.numNodes

        avgDeg = 0  # The average node degree of the graph

        # Building the xpoints and ypoints list
        for i in range(self.numNodes):
            xpoints[i] = i
            if i in degreeDist:
                avgDeg += i * degreeDist[i]  # Adding (degree * weight of degree)
                # Fraction of nodes in the graph with that degree
                ypoints[i] = degreeDist[i] / self.numNodes

        # avgDeg = sum(degree * weight of degree) / total_nodes
        avgDeg /= self.numNodes

        # Plotting the average node degree as a vertical line
        plt.axvline(x=avgDeg, color="r", label="Avg. node degree")

        # Plotting the curve for (fraction of nodes vs Node degree)
        # Giving it a lower z-index to push them behind the grid lines
        plt.plot(xpoints, ypoints, "ob", label="Actual degree distribution", zorder=0)

        # Plotting the grid lines with higher z-index that the curve
        plt.grid(zorder=1)

        plt.legend()  # Displaying the legend box

        plt.show()  # Displaying the graph

    @handleError
    def _isConnectedHelper(self, sv, visited):
        """
        BFS Helper function that runs BFS from a given starting node.\n
        `visited` list is to keep track of which all nodes are visited till now in the BFS
        """
        q = Queue()  # queue data structure
        q.put(sv)  # Adding the starting vertex to the queue
        visited[sv] = True  # Marking the starting vertex as visited
        while not q.empty():  # While the queue is not empty
            u = q.get()  # Pop the first element from the queue
            for v in self.gr[u]:  # Visit all the neighbours of the popped out node
                if not visited[v]:  # If that neighbour is not visited
                    q.put(v)  # Push it into the queue
                    visited[v] = True  # Mark it as visited

    @handleError
    def isConnected(self):
        """
        Checks connectedness of the graph.\n
        Returns True if the graph is connected, and False otherwise.
        """
        # Initializing the `visited` list for the graph nodes
        visited = [False] * (self.numNodes + 1)
        numComponents = 0  # Number of connected components of the graph
        # Looping all the nodes of the graph
        for i in range(1, self.numNodes + 1):
            if not visited[i]:  # If the node is not visited, start BFS from it
                self._isConnectedHelper(i, visited)  # Executing BFS from the found node
                numComponents += 1  # Incrementing the number of connected components by
                if numComponents > 1:  # If the number of connected components is > 1
                    return False
        return True


class ERRandomGraph(UndirectedGraph):
    """
    Class derived from Undirected Graph Class to create a Erdos-Renyi random graph G(n, p)
    """

    def __init__(self, n):
        """
        Constructor for the class, takes as input the number of nodes in the graph
        """
        super().__init__(n)  # Calling the parent class constructor

    @handleError
    def sample(self, p):
        """
        Generates a random graph G(n, p) for the input probability p
        Loops through all the edges of the graphs possible and generates a random number for each edge from [0, 1) and if that number is less than p, then we add the edge b/w those two nodes.
        """
        # Re-initializing the graph
        super().__init__(self.numNodes)

        # Looping though all the edges
        for i in range(1, self.numNodes + 1):
            for j in range(i + 1, self.numNodes + 1):
                # If random number generated is less than p
                if random() < p:
                    # Add the edge b/w the two graphs
                    self.addEdge(i, j)

    @handleError
    def verifyERConnectednessStatement(self):
        """
        Verifies and visualizes the Erdos-Renyi model that the graph G(n, p) is almost surely connected only if `p > log(n) / n`
        """
        # The theoretical threshold which is `log(n) / n`
        theoretical_threshold = log(self.numNodes) / self.numNodes
        numRuns = 1000  # Number of runs executed for each probability
        minP = 0  # Minimum probability
        maxP = theoretical_threshold * 3  # Just a random number
        # Generating probability points range from [minP, maxP]
        numProbPoints = 50  # Number of probability points generated
        prob = list(linspace(minP, maxP, numProbPoints))
        # fraction of runs G(n, p) is connected for each probability
        fracConnected = []
        for p in prob:
            count = 0  # Number of G(n, p) found as connected
            # Generating G(n, p) `numRuns` times
            for _ in range(numRuns):
                self.sample(p)  # Sampling a random graph
                if self.isConnected():  # Checking if it is connected
                    count += 1  # Incrementing the count

            # fraction of runs G(n, p) is connected
            fracConnected.append(count / numRuns)

        # Giving title and labels to the plot
        plt.title(
            "Connectedness of a G({n}, p) as function of p".format(n=self.numNodes)
        )
        plt.xlabel("p")
        plt.ylabel("fraction of runs G({n}, p) is connected".format(n=self.numNodes))

        # Removing the margins from the y-axis
        plt.margins(y=0)

        # Plotting the curve for `fracConnected vs prob`
        plt.plot(prob, fracConnected, color="b")

        # Plotting the theoretical threshold as a vertical line
        plt.axvline(x=theoretical_threshold, color="r", label="Theoretical threshold")
        # Plotting the grid lines
        plt.grid()
        # Plotting the legend box
        plt.legend()
        # Displaying the curve
        plt.show()


if __name__ == "__main__":
    # Sample Test Case 1
    g = UndirectedGraph(5)
    g = g + (1, 2)
    g = g + (2, 3)
    g = g + (3, 4)
    g = g + (3, 5)
    print(g.isConnected())

    # Sample Test Case 2
    g = UndirectedGraph(5)
    g = g + (1, 2)
    g = g + (2, 3)
    g = g + (3, 5)
    print(g.isConnected())
    print(g)

    # Sample Test Case 3
    g = ERRandomGraph(100)
    g.verifyERConnectednessStatement()
    # The above plot shows correctedness average over 1000 runs for 50 number of probability points on x-axis
    # Execution Time on my PC: 3 minutes approx.
