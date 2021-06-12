"""
Imports
"""
import time, enum, math
import numpy as np
import pandas as pd
import pylab as plt
from mesa import Agent, Model
from mesa.time import SimultaneousActivation, RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from networkx.algorithms.shortest_paths.generic import has_path
import networkx as nx
import panel as pn          
import panel.widgets as pnw
import random
import pickle
import csv
from tqdm import tqdm, trange
from time import sleep

data_path = '../' #set to wherever the data files are, will be used on every input

ports = pd.read_csv((data_path +'ports.csv'))

#Loading Files
origin = pd.read_csv((data_path + 'origin_ports.csv'))
data = pd.read_csv((data_path + 'clean_distances.csv')) # i keep the old name
distances = data[["prev_port", "next_port", "distance"]] 
distances.astype({'prev_port':'int64', 'next_port':'int64'}).dtypes
origin = origin.astype({'Ref':'int64'})


#Route Blockage Import
route_blockage_dov = pd.read_csv((data_path + 'route_blockages_dov.csv'))
route_blockage_gib = pd.read_csv((data_path + 'route_blockages_gib.csv'))
route_blockage_horm = pd.read_csv((data_path + 'route_blockages_horm.csv'))
route_blockage_mal = pd.read_csv((data_path + 'route_blockages_mal.csv'))
route_blockage_pan = pd.read_csv((data_path + 'route_blockages_pan.csv'))
route_blockage_suez = pd.read_csv((data_path + 'route_blockages_suez.csv'))
route_blockage_total = pd.read_csv((data_path + 'route_blockages_total.csv'))
pruning_files = [route_blockage_dov, route_blockage_gib, route_blockage_horm, route_blockage_mal, route_blockage_pan, route_blockage_suez, route_blockage_total]




















"""
Output
"""

# write output of batch runner to file
with open((data_path + 'test.pickle'), 'wb') as handle:
    pickle.dump(data_collector_agents, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(( data_path + 'keys.csv'), 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for a, b, c in list(keys):
       writer.writerow([a, b, c])




"""
Load files for analysis
"""

# with open((data_path + 'test.pickle'), 'rb') as handle:
#     b = pickle.load(handle)

# with open(( data_path + 'keys.csv')) as csv_file:
#     reader = csv.reader(csv_file)
#     mydict = list(reader)