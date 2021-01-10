import os 
import sys
import time
import numpy as np
import random
import itertools as it
import math
from itertools import combinations
import pandas as pd
import copy
import pickle as p

# there are two types of input file to be dealt with
# 1. tsp; 2. GEO

def read_data(input_name):

    '''
    e.g: data = read_data(input data file)
    args: input data file
    Return: Nx2 matrix: each row: [x, y]
    '''
    data = []
    with open(input_name, "r") as file:
        lines = file.readlines()
        
        for i in range(len(lines)):
            # deal with different input
            line = lines[i].lstrip()
            if line:
                if line[0].isdigit():
                    node_id, x, y = line.split()
                    data.append([float(x),
                                 float(y)])
    return data
	
	
class VNS:
    def __init__(self, tsp,mutation = False,locus=False,max_attempts=20, neighbourhood_size=5):
        # distance input: node_id, x, y
        self.locus = locus
        self.mutation = mutation
        self.tsp = tsp
        self.max_attempts = max_attempts
        self.neighbourhood_size = neighbourhood_size
        self.ProblemSize = len(self.tsp)
        self.best_tour = None 
        self.best_cost = float("inf") 
        self.nodes = [i for i in range(self.ProblemSize)]
        self.GenerateProblem()
        self.temp_tour = None
        
    def GenerateProblem(self):
        OrigDistanceMatrix=np.zeros((self.ProblemSize,self.ProblemSize))
        for i in range(len(self.tsp)):
            for j in range(i+1,len(self.tsp)):
                OrigDistanceMatrix[i,j] = np.sqrt((self.tsp[i][0]-self.tsp[j][0])**2
                                                  +(self.tsp[i][1]-self.tsp[j][1])**2)
                OrigDistanceMatrix[j,i] = OrigDistanceMatrix[i,j] + 0.05

        #put infinite values into the diagonal
        self.DistanceMatrix= np.copy(OrigDistanceMatrix)
        self.MaxDistanceFromCity= np.max(self.DistanceMatrix,0)
        self.DistanceMatrix = OrigDistanceMatrix
        self.DistanceMatrix+= np.where(np.eye(OrigDistanceMatrix.shape[0])>0,np.inf,0)
        self.MinDistanceFromCity= np.min(self.DistanceMatrix,0)
        self.NormalizationFactor=(self.MaxDistanceFromCity-self.MinDistanceFromCity)
    
    def Fittnes(self,Vector):
        #this funciton calculates the total distance traveled between citie
        LocusWeights=np.zeros(self.ProblemSize)
        LocusWeights[0]=0.1
        for i in range(self.ProblemSize-1):
            LocusWeights[i+1]=((self.DistanceMatrix[int(Vector[i]),int(Vector[i+1])]
                                -self.MinDistanceFromCity[int(Vector[i+1])])/self.NormalizationFactor[int(Vector[i+1])])+0.1  
        #the first city mutation factor is the average weight   
        return LocusWeights
    
    
    def greedy(self):
        cur_node = random.randint(0, self.ProblemSize-1)
        solution = [cur_node]

        remain_nodes = set(self.nodes)
        remain_nodes.remove(cur_node)

        while remain_nodes:
            next_node = min(remain_nodes, key=lambda x: self.DistanceMatrix[cur_node][x])
            remain_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node

        cur_total_dis = self.get_total_dist(solution)
        if cur_total_dis < self.best_cost:
            self.best_cost = cur_total_dis
            self.best_tour = solution
        return solution, cur_total_dis
    
    def get_total_dist(self, tour):
        cur_total_dis = 0
        for i in range(self.ProblemSize-1):
            cur_total_dis += self.DistanceMatrix[tour[i]][tour[(i + 1)]]
        return cur_total_dis

    # Function: Stochastic 2_opt
    def stochastic_2_opt(self,city_tour):
        curr_route = city_tour
        '''        
        i, j  = random.sample(range(0, len(city_tour)), 2)
        if (i > j):
            i, j = j, i
        curr_route[i:j+1] = list(reversed(curr_route[i:j+1]))
        '''
        best_route = copy.deepcopy(curr_route)
        if self.mutation:
            if self.locus:
                self.temp_tour=curr_route
                LocusWeights = self.Fittnes(curr_route)
                LocusWeights /= LocusWeights.sum()
                Indices=np.random.choice(len(city_tour), 2, replace=False,p=LocusWeights)
                best_route[Indices[0]]=curr_route[Indices[1]]
                best_route[Indices[1]]=curr_route[Indices[0]]

            else:     
                Indices=np.random.choice(len(city_tour), 2, replace=False)
                best_route[Indices[0]]=curr_route[Indices[1]]
                best_route[Indices[1]]=curr_route[Indices[0]]
        return best_route


    # Function: Local Search
    def local_search(self,city_tour):
        count = 0
        solution = city_tour 
        while (count < self.max_attempts): 
            for i in range(0, self.neighbourhood_size):
                candidate = self.stochastic_2_opt(solution)
            candidate_cost = self.get_total_dist(candidate)
            solution_cost = self.get_total_dist(solution)
            if candidate_cost < solution_cost:
                solution  = candidate
                count = 0
            else:
                count = count + 1                             
        return solution 

    def variable_neighborhood_search(self,curr_tour = None, iterations = 50):
        count = 0
        costs=[]
        self.best_tour = curr_tour
        self.best_cost = self.get_total_dist(curr_tour)
        while (count < iterations):
            for i in range(0, self.neighbourhood_size):
                for j in range(0, self.neighbourhood_size):
                    new_tour = self.stochastic_2_opt(self.best_tour)
                new_tour = self.local_search(new_tour)
                new_tour_cost = self.get_total_dist(new_tour)
                if (new_tour_cost < self.best_cost):
                    self.best_tour = new_tour
                    self.best_cost = new_tour_cost
                    break
            count = count + 1
            costs.append(self.best_cost)
        return costs 

files = []
for file in os.listdir('./routs'):
    points = read_data('./routs/' + file)
    if len(points)<101:
        files.append(file)

repeat = 10
iteration = 100

max_attempts=30
neighbourhood_size=5

#Original_costs= np.zeros((len(files),repeat,iteration))
Random_Costs = np.zeros((len(files),repeat,iteration))
Locus_Costs = np.zeros((len(files),repeat,iteration))

for k in range(len(files)):
    points = read_data('./routs/' + files[k])
    for i in range(repeat):  
        MVNS = VNS(points,max_attempts=max_attempts, neighbourhood_size=neighbourhood_size,mutation=True)
        curr_tour, _ = MVNS.greedy()
        Random_Costs[k][i] = MVNS.variable_neighborhood_search(curr_tour,iterations = iteration)

        LVNS = VNS(points,max_attempts=max_attempts, neighbourhood_size=neighbourhood_size,mutation=True,locus=True)
        #curr_tour, _ = LVNS.greedy()
        Locus_Costs[k][i] = LVNS.variable_neighborhood_search(curr_tour,iterations = iteration)
    print(files[k] ,' costs')
    #print('Original: ', np.mean(Original_costs[k][:][-1]))
    print('Random: ', np.mean(Random_Costs[k][:][-1]))
    print('Locus: ', np.mean(Locus_Costs[k][:][-1]))
 

p.dump(Random_Costs,open('VNS_Random_costs.p','wb'))
p.dump(Locus_Costs,open('VNS_Locus_costs.p','wb'))