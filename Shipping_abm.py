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
from mesa.batchrunner import BatchRunnerMP
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

#Our Schedule (run over 90 days ~ 3 months, we are interested in the behavior shortly before and after the network changes) is the 
# pruning_schedule = [{30:"Dover", 45:"Open"}, {30:"Gibraltar", 45:"Open"}, {30:"Hormuz", 45:"Open"}, {30:"Malacca", 45:"Open"}, {30:"Panama", 45:"Open"}, {30:"Suez", 45:"Open"}, {30:"Total", 45:"Open"}]
pruning_schedule = ["Dover", "Gibraltar", "Hormuz", "Malacca", "Panama", "Suez", "Total"]
pruning_schedule_single = {30:"Dover", 45:"Open"}


"""
Model
"""
class ShippingNetwork(Model):
    def __init__(self, distances, major_ports, pruning_files, pruning_schedule, S=100,  s=20, f = 0, x = 3):
        self.major_ports = major_ports
        self.num_ships = S
        self.distances = distances
        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.Ships = []
        self.pruning_files = pruning_files
        self.s = s
        self.f = f
        self.pruning_schedule = pruning_schedule
        self.stp_cnt = 0
        self.x = x 

        #Build Network without closures
        self.G = nx.from_pandas_edgelist(distances, "prev_port", "next_port", ["distance"], create_using=nx.Graph())
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        #Define Mesa Grid as the just created Network to allow for shipping only in routes
        self.grid = NetworkGrid(self.G) 

        #Build alternate Networks (with closures in place)
        self.G_Dov = self.Cut_Graph(self.G, self.pruning_files[0])
        self.G_Gib = self.Cut_Graph(self.G, self.pruning_files[1])
        self.G_Horm = self.Cut_Graph(self.G, self.pruning_files[2])
        self.G_Mal = self.Cut_Graph(self.G, self.pruning_files[3])
        self.G_Pan = self.Cut_Graph(self.G, self.pruning_files[4])
        self.G_Suez = self.Cut_Graph(self.G, self.pruning_files[5])
        self.G_Total = self.Cut_Graph(self.G, self.pruning_files[6])
  


        #create agents 
        Ships = []
       
        for i in tqdm(range(self.num_ships), desc="Placing Ships"):
        
            a = Ship(i+1, self, self.G, self.major_ports,  self.s, self.f, self.x)
            self.schedule.add(a)
            #append to list of ships
            Ships.append(a)
        
            #place agent on origin node
            self.grid.place_agent(a, a.start)



        self.datacollector = DataCollector(
            model_reporters={"Graph":"blockage"},
            agent_reporters={"Type": "ship_class",
                            "Foresight": "foresight",
                            "Position": "position", 
                            "Ports of Interest":"report_ports", 
                            "Itinerary":"itinerary", 
                            "Distance_Traveled":"distance_traveled", 
                            "Route":"current_route", 
                            "Route Changes":"route_chng", 
                            "Destination not reachable" : "not_reachable",
                            "Complete": "complete_route",
                            "Success": "sucess", 
                            "Stuck":"stuck"})

        '''
        Ennsure usage of correct Cut Graph Method
        '''

    def Cut_Graph(self, G, route_blockages):
        return_G = G.copy()  #CRUCIAL TO INCLUDE COPY STATEMENT
        for index in range(len(route_blockages)):
            try:
                return_G.remove_edge(route_blockages.iloc[index]['prev_port'],route_blockages.iloc[index]['next_port'])
            except:
                pass
        return return_G


    '''
    Method allows for change of network (copies specified pre-built )
    '''
        #create ability to remove edges mid-model
    def network_change(self, blockage):
        if blockage == "Dover":
            G_new = self.G_Dov
        elif blockage == "Gibraltar":
            G_new = self.G_Gib
        elif blockage == "Hormuz":
            G_new = self.G_Horm
        elif blockage == "Malacca":
            G_new = self.G_Mal
        elif blockage == "Panama":
            G_new = self.G_Pan
        elif blockage == "Suez":
            G_new = self.G_Suez
        elif blockage == "Total":
            G_new = self.G_Total
        elif blockage == "Open":
            G_new = self.G

        for a in tqdm(self.Ships):
                a.G = G_new

    # def network_check(self):
    #     try:
    #         self.network_change(self.pruning_schedule[self.stp_cnt])
    #     except:
    #         pass

    def step(self):
        if self.stp_cnt == 30:
            self.network_change(self.pruning_schedule)
        elif self.stp_cnt == 60:
            self.network_change("Open")
        # self.network_check()    #Check if it is time to change the network as per the schedule
        self.schedule.step()    #Run each Agents
        self.datacollector.collect(self)
        self.stp_cnt += 1



"""
Ship Agents
"""
class Ship(Agent):
    def __init__(self, unique_id, model, G, major_ports, s, f, x):
        super().__init__(unique_id, model)
        
        self.major_ports = major_ports
        self.G = G
        self.s = s
        self.f = f
        self.x = x
        self.factor = 1 * self.x

        self.ship_class = np.random.choice(["Large","Normal", "Small"], 1, p=[0.5, 0.25, 0.25])

        self.start_port = self.origin()
        
        self.destination = self.dest() #sample a destination
        #We sample the origin port from a list of the 50 biggest ports world, with the prob = TAU of the port / TAU of all origin ports for 2017
        self.ports =  [*self.start_port, *self.destination]
        self.report_ports = self.ports.copy()
        
        self.foresight = np.random.poisson(self.f)

        self.state = 0 #0 for active, numbers > 0 for weeks that ships have to "wait" until arrival to port
        self.speed = self.s*24*1.852 #speed is given in knots, with 1 knot being 1 nautical mile per hour. Since the model works with distances in km, we convert here (1 nm = 1.852m)

        self.not_reachable = 0 ##global counter for Networx error

        self.origin_failed = 0 #counter for ships not able to reach any of the ports

        self.init_route, self.init_dist = self.routing() #We keep a copy of the entire itinerary / distance traveled
        self.init_dist = round(self.init_dist,2)
        self.start = self.start_port[0]
        self.current_route, self.current_dist = self.init_route.copy(), self.init_dist  #For comparison & navigational purposes, we use current route & distance
        self.start_speed = self.speed # varying speed
        self.position = self.current_route[0]
        self.next_position = self.current_route[1]
        self.target  = round((self.init_dist // self.start_speed) * self.factor,1) #target to reach all destinations
        self.itinerary = [self.position]
        self.distance_traveled = 0
        self.unique_id = unique_id
        self.step_size = self.ident_distance()
        self.route_chng = 0
        self.complete_route = 0
        self.steps = 0
        self.sucess = 0
        self.stuck = 0

    def origin(self):
        """
        Sample origin based on ship type.
        """ 
        
        if self.ship_class == "Large":
            start_port = np.random.choice(self.major_ports["Ref"],  p=self.major_ports["PROB"])
        elif self.ship_class == "Normal":
            start_port = np.random.choice(self.major_ports["Ref"],  p=self.major_ports["PROB"])

        else:
            r = random.sample(self.G.nodes, k=1)[0] #ships do not originate in isolated nodes
            if not nx.is_isolate(self.G,r):
                start_port = r
            else:
                return origin()
            
        return [start_port]
        

    def dest(self):
        """
        Sample destinations
        """
        if self.ship_class == "Large": #large ships only visit large ports

            #we try to mix in top 10 ports with the rest

            p1 = self.major_ports["PROB"][:10].copy()
            p1 /= p1.sum()
            p2 = self.major_ports["PROB"][10:].copy()
            p2 /= p2.sum()
            k = np.random.randint(1, high = 3)
            end = np.random.choice(self.major_ports["Ref"][:10] , size=k,  p=p1).tolist() + np.random.choice(self.major_ports["Ref"][10:], size=k,  p=p2).tolist()
                        
            
        elif self.ship_class == "Normal":
            k = np.random.randint(1, high = 4)
            end = np.random.choice(self.major_ports["Ref"], replace=False, size=k,  p=self.major_ports["PROB"]).tolist()+ [float(i) for i in random.sample(self.G.nodes, k=k)]

        else:
            k = np.random.randint(1, high =6)
            end = [int(i) for i in random.sample(self.G.nodes, k=k)]

        return end


    def routing(self):
        """
        A greedy version of Travelling Salesman algorithm.
        Takes in a list of ports, with the first port being the origin.
        It loops to find the closest port. Returns a list of ports to visit (an itinerary) and the overall distance.
        """
        ports = self.ports.copy()
        overall_distance = list()
        itinerary = list() 
        itinerary.append([ports[0]])
        not_reached = 0 #local counter for no path between points 
        
        for j in range(len(ports)):
            try:
                distance = dict()
                
                for i in range(1,len(ports)): #look for the closest port
                    try:
                        distance[ports[i]] = nx.shortest_path_length(self.G, ports[0] , ports[i], weight='distance')
                    except nx.NetworkXNoPath:
                        self.not_reachable += 1 #global counter for Networx error
                        not_reached += 1 #local counter
                        return routing()

                    except AttributeError:
                        continue
                next_stop = min(distance, key=distance.get)
                itinerary.append(nx.shortest_path(self.G, ports[0], next_stop, weight = 'distance')[1:]) #add the route to the closest port to the itinerary
                overall_distance.append(distance.get(next_stop)) #add distance to the closest port
                ports.pop(0)
                ind = ports.index(next_stop)
                ports.pop(ind)
                ports.insert(0, next_stop)



            except ValueError: #handle list end
                pass

        try:    
            for j in range(len(ports)):
                distance = dict()
                try:
                    for i in range(1,len(ports)): #look for the closest port
                        distance[ports[i]] = nx.shortest_path_length(self.G, ports[0] , ports[i], weight='distance')
                        next_stop = min(distance, key=distance.get)
                        itinerary.append(nx.shortest_path(self.G, ports[0], next_stop, weight = 'distance')[1:]) #add the route to the closest port to the itinerary
                        overall_distance.append(distance.get(next_stop)) #add distance to the closest port
                        ports.pop(0)
                        ind = ports.index(next_stop)
                        ports.pop(ind)
                        ports.insert(0, next_stop)

                except nx.NetworkXNoPath:
                    self.not_reachable += 1 #global counter for Networx error
                    not_reached += 1 #local counter

        except ValueError: #handle list end
                pass

        if not_reached == len(ports): #if no routes possible routes found, reassign destination
            self.origin_failed += 1
            self.destination = self.dest()
            self.ports =  [*self.start_port, *self.destination]
            if self.origin_failed > 1: #if the problem persists, the ship is stuck
                self.stuck += 1
            else:
                return self.routing() #recursion brrrrr
        else:   
            flat_route = []
            for sublist in itinerary: #flatten the itinerary
                for port in sublist:
                    flat_route.append(port)
            travel_distance = sum(overall_distance)
            return flat_route, travel_distance

    def move(self):

        self.step_size = self.ident_distance() #look up the distance between two cities 
        self.state = self.step_size / self.speed #change state to step amount
        self.current_dist = self.current_dist - self.step_size #adjust current distance minus the distance traveled in the next step
        self.model.grid.move_agent(self, self.next_position) #move the agent
        self.position = self.next_position
        if self.position in self.destination:
            self.destination.pop(self.destination.index(self.position))
        
        self.current_route.pop(0) #remove the next step from the itinerary
        # self.position = self.next_position
        if len(self.current_route) == 1:
            self.next_position = self.current_route[0] 
        else:
            self.next_position = self.current_route[1] #update current route


    def ident_distance(self): #look up the distance of the current step
        try:
            return round(self.G.get_edge_data(self.position, self.next_position, default=0)['distance'],2)
        except:
            return 0
    
    def new_destinations(self): #the ship has completed its full route
        
        self.complete_route += 1
        self.destination = self.dest()
        self.ports =  [self.position, *self.destination]
        self.report_ports = self.ports.copy()
        self.init_route, self.init_dist = self.routing()
        self.init_dist = round(self.init_dist,2)
        self.current_route, self.current_dist = self.init_route.copy(), self.init_dist
        self.target  = (self.init_dist // self.start_speed)  * self.factor
        self.state =  np.random.randint(2,5) #wait at port
 
    def step(self):
        self.state = self.state - 1 #'move' ships by one day progress
        if self.stuck >= 1:
            pass
        else:
            if round(self.state,2) <= 0: #ships that are en-route to the node they are going to next do not move / perform other activities
                self.distance_traveled += self.step_size #ship has arrived at port, let's add the distance traveled to their 
            
                #add the current position to itinerary
                if self.position != self.current_route[-1]: #if current stop is not the final stop
                    self.ports =  [self.position, *self.destination]
                    new_route, new_distance = self.routing() #perform a new routing to compare against current routing
                    new_distance= round(new_distance,2)
                    if (new_route == self.current_route)| (new_distance == self.current_dist): #if current routing is the same as new, just move (default case)
                        # print("default case")
                        self.move()
                        self.itinerary.append(self.position)
                        self.steps += 1 
            
                    # Compare the distances
                    elif new_distance > self.current_dist: #if current route is shorter than newly calculated route, check for obstructions
                        
                        if self.foresight >= (self.current_dist // self.speed): #check how many steps are you from your final destination. If you are far away, do nothing and remain on course
                            # if not has_path(self.G, self.position, self.next_position): 
                            self.current_route = new_route
                            self.current_dist = new_distance
                            self.route_chng += 1
                            self.move()
                            self.itinerary.append(self.position)
                            self.steps +=1
                        else:
                            self.move()
                            self.itinerary.append(self.position)
                            self.steps +=1
                    
                    
                    else: # final option is that current route is longer than new route (think Suez reopening after a while), here, we just take the new option
                        self.current_route = new_route
                        self.current_dist = new_distance
                        self.route_chng += 1
                        self.move()
                        self.itinerary.append(self.position)
                        self.steps +=1
                
                else: #if ship is arrived at final position, get a new route, and start back
                    if self.steps >= self.target: #if the ship manages the reach all the destinations in time, it is "sucessful"
                        self.sucess += 1
                        self.steps = 0
                        self.new_destinations()
                        self.itinerary.append(self.position)
                         #clear the counter
                    else:
                        self.new_destinations()
                        self.itinerary.append(self.position)
                        self.steps = 0 #clear the counter
            else:
                pass




"""
Model Instantiation & Output
"""

#Single Run
# model = ShippingNetwork(distances, origin, pruning_files, pruning_schedule_single, 100)


# for i in trange(steps):
#     model.step()


# agent_state = model.datacollector.get_agent_vars_dataframe()


# #write output of single run to file
# agent_state.to_csv((data_path + 'single_run_output.csv'), header = True)



#Multiple runs using Batchrunner
fixed_params = {"distances": distances, "major_ports":origin, "pruning_files": pruning_files, "S": 5}
variable_params = {"f": range(0, 20, 5), "x": np.arange(-0.75, 1.25, 0.25), "pruning_schedule": pruning_schedule }

batch_run = BatchRunnerMP(ShippingNetwork,
                        nr_processes=5,
                        variable_prams = variable_params,
                        fixed_params = fixed_params,
                        iterations=3,
                        max_steps=90,
                        )
batch_run.run_all()


data_collector_agents = batch_run.get_collector_agents()
keys = data_collector_agents.keys()









# write output of batch runner to file
with open((data_path + 'batch_out.pickle'), 'wb') as handle:
    pickle.dump(data_collector_agents, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(( data_path + 'batch_keys.csv'), 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for a, b, c, d in list(keys):
       writer.writerow([a, b, c, d])    




"""
Load files for analysis
"""

# with open((data_path + 'test.pickle'), 'rb') as handle:
#     b = pickle.load(handle)

# with open(( data_path + 'keys.csv')) as csv_file:
#     reader = csv.reader(csv_file)
#     mydict = list(reader)