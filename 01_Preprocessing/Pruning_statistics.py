import time, enum, math
import numpy as np
import pandas as pd
import pylab as plt
import csv
from networkx.algorithms.approximation.connectivity import node_connectivity
from networkx.algorithms.shortest_paths.generic import has_path
import networkx as nx
import random
from tqdm import tqdm, trange

data_path = '../'

"""
Imports
"""
clean_distances = pd.read_csv((data_path + 'clean_distances.csv'))
route_blockage_dov = pd.read_csv((data_path + 'route_blockages_dov.csv'))
route_blockage_gib = pd.read_csv((data_path + 'route_blockages_gib.csv'))
route_blockage_horm = pd.read_csv((data_path + 'route_blockages_horm.csv'))
route_blockage_mal = pd.read_csv((data_path + 'route_blockages_mal.csv'))
route_blockage_pan = pd.read_csv((data_path + 'route_blockages_pan.csv'))
route_blockage_suez = pd.read_csv((data_path + 'route_blockages_suez.csv'))
route_blockage_total = pd.read_csv((data_path + 'route_blockages_total.csv'))

pruning_files = [route_blockage_dov, route_blockage_gib, route_blockage_horm, route_blockage_mal, route_blockage_pan, route_blockage_suez, route_blockage_total]
pruning_names = ["Dover", "Gibraltar", "Hormuz", "Malacca", "Panama", "Suez", "Total"]


"""
Building Graphs
"""
G = nx.from_pandas_edgelist(clean_distances, "prev_port", "next_port",edge_attr= "distance",create_using=nx.Graph())

def Cut_Graph(G, route_blockages):
        return_G = G.copy()
        fails = 0
        for index in range(len(route_blockages)):
            try:
                return_G.remove_edge(route_blockages.iloc[index]['prev_port'],route_blockages.iloc[index]['next_port'])
            except:
                fails += 1
        print("Failed {} times".format(fails))
        return return_G


"""
Running Descriptive Collector
"""
results = [["File", "Counter", "Success", "Distance Difference", "Same Distance", "Fail", "Note Connectivity Unchanged", "Node Connectivity Changed"]]
for i in trange(len(pruning_files),desc="Running Descriptive Collector"):
    file = pruning_files[i]
    G_changed = Cut_Graph(G, file)

    distance_diff = 0
    success = 0
    fail = 0
    same_distance = 0
    counter = 0
    connect = 0
    changed_connect = 0
    


    for k in trange((30), desc="Running for File" ):
        position = list(G.nodes())[k]
        for j in trange((30), desc="Running for Position"):
            destination = list(G.nodes())[j]
            try:
                init_dist = nx.dijkstra_path_length(G, position, destination, weight='distance')

                changed_dist = nx.dijkstra_path_length(G_changed, position, destination, weight='distance')
                if init_dist != changed_dist:
                    
                    distance_diff += changed_dist - init_dist
                    success += 1
                    counter += 1
                    connect += node_connectivity(G, s=position, t=destination)
                    changed_connect += node_connectivity(G_changed, s=position, t=destination)
                else:
                    same_distance += 1
                    counter += 1
                    connect += node_connectivity(G, s=position, t=destination)
                    changed_connect += node_connectivity(G_changed, s=position, t=destination)
            except :
                fail += 1
                counter += 1
    result = [pruning_names[i], counter, success, distance_diff, same_distance, fail, connect, changed_connect]
    results.append(result)

with open('Pruning_Stats.csv', 'w') as f: 
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(results)

