import networkit as nk
import matplotlib.pyplot as plt

class Graph():
    def __init__(self, test_list) -> None:
        self.tests = test_list

    def plot_tests_vs_result_vars(self):
        G = nk.Graph(10, directed=True, weighted=True)

                #Add edges to the graph
        G.addEdge(1, 3)
        G.addEdge(2, 4)
        G.addEdge(1, 2)
        G.addEdge(3, 4)
        G.addEdge(2, 3)
        G.addEdge(4, 0)
        #Set weights to edges
        G.setWeight(1, 3, 2)
        G.setWeight(2, 4, 3)
        G.setWeight(3, 4, 4)
        G.setWeight(4, 0, 5)
        #To see the graph structural overview
        print(nk.overview(G))
        #For visualization
        nk.viztasks.drawGraph(G)
        plt.show()
