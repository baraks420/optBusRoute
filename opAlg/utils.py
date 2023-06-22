import numpy as np
from pyvis.network import Network
from matplotlib import pyplot as plt

no=np.inf

def plot_best_route(G,exp,best_path,nb_comb,alpha,sum_exp):
    """
    Parameters:
    - G : Grpah matrix with distance between stations
    - exp : List of expectation of each station
    - best_path : The best route found by the algoritm
    """
    
    # Create a new network instance
    title= "routes/iter :"+str(nb_comb)+" - "+"alpha :"+str(alpha)+" - Sum of exp : "+str(sum_exp)
    net = Network(notebook=True, heading=title)

    # Add nodes
    num_nodes = G.shape[0]
    
    for i in range(num_nodes):
        if i==0:
            net.add_node(i, label=f'{i}', color='#22b512', title='Expected passenger: '+str(0), size=10)
        else:
            net.add_node(i, label=f'{i}', color='#22b512', title='Expected passenger: '+str(exp[i-1]), size=10)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if G[i,j] not in [no,0]:
                dist = G[i,j]
                net.add_edge(i, j, color="blue", title=f'{dist}', length=dist)

    # Define the path to be colored
    path=[]
    path.append((0,best_path[0]))
    i=0
    while i<len(best_path):
        if i+1 < len(best_path):
            path.append((best_path[i],best_path[i+1]))
        i+=1
    path.append((best_path[-1],0))
        
    # Set color for the path edges
    for edge in net.edges:
        if (edge['from'], edge['to']) in path or (edge['to'], edge['from']) in path:
            edge['color'] = 'red'

    # Visualize the graph network
    net.show_buttons(filter_=['physics'])

    net.show('graph.html')
    


def plot_iteration(generation,nb_comb,alpha):
    """
    input :
    - List of all generations
    """
    max_exp=[]
    min_exp=[]
    mean_exp=[]
    for i in range(len(generation)):
        max_exp.append(generation[i]["expectation"].max())
        min_exp.append(generation[i]["expectation"][generation[i]["expectation"] != -1].min())
        mean_exp.append(generation[i]["expectation"].mean())
    
    
    plt.plot(range(len(generation)),max_exp, color='green', label="Best route cost")
    #plt.plot(range(len(generation)),min_exp, color='red', label="min expectations")
    #plt.plot(range(len(generation)),mean_exp, color='blue', label="mean expectations")
    plt.title("Best Route cost by iteration : "+str(nb_comb)+" routes/iter & aplha = "+str(alpha))
    plt.xlabel("iterations")
    plt.ylabel("best route cost")

    plt.legend()
    plt.show()
    



