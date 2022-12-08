import sys
import matplotlib.pyplot as plt
import networkx as nx

NNODES = 10
SEED = 12345

def main():
    """Main function"""

    # Generate original graph
    OGnx = nx.complete_graph(NNODES)  # TODO: need to add EDGE WEIGHTS!
    # Generate node positions
    pos = nx.drawing.layout.random_layout(OGnx, seed=SEED)

    # Compute MST
    myMST = nx.minimum_spanning_tree(OGnx)

    stop = 1

    if 0:
        # Convert to array
        myMSTarray = myMST.toarray().astype(int)    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html

        # Convert to networkx object
        MSTnx = nx.Graph(myMSTarray)

        # Plot original graph
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(5, 5)
        nx.draw(OGnx, pos=pos, ax=axes)
        nx.draw(MSTnx, pos=pos, ax=axes, edge_color='r', width=2.0)
        # axes.legend()
        axes.set_title('Minimum-Span Tree w/ scipy.sparse.csgraph._min_spanning_tree')
        plt.show()
        stop = 1


if __name__ == '__main__':
    sys.exit(main())