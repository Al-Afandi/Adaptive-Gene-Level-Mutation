import os 
import sys
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools as it
import os
import csv

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
    
    
#check all files
for file in os.listdir('./ALL_tsp'):
    tsp = read_data('./ALL_tsp/' + file)
    #print(file,'  ',len(tsp))
    
    
    
    
    
class GATest():
    def __init__(self,tsp,S=32,P=200,I=50):
        #problem parameters
        self.ProblemSize=S #number of cities
        self.PopulationSize=P
        self.NumIteration=I
        self.tsp = tsp
        self.DistanceMatrix=np.zeros((self.ProblemSize,self.ProblemSize))

        #parameters of the algorithm
        self.KeepRatio=0.05 #this proportion of the population will be kept
        self.NewRatio=0.2 #this proportion iwll be newly generated
        self.CrossoveredRatio=1-(self.KeepRatio+self.NewRatio) #this proprtion will be created as new offsprings
        self.MutationRate=0.5 #probability of mutation of an instance
        self.Population=np.zeros((self.PopulationSize,self.ProblemSize)) #population is stored in this table
        self.Weights=np.zeros(self.PopulationSize) 
        self.MinWeights=np.zeros(self.NumIteration)
        self.GenerateProblem()
        self.ReInitPopulation()
    
    def setter(self,MutationRate):
        self.MutationRate=MutationRate #probability of mutation of an instance
        
    def GenerateProblem(self):
        OrigDistanceMatrix=np.zeros((self.ProblemSize,self.ProblemSize))
        for i in range(len(tsp)):
            for j in range(i+1,len(tsp)):
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
        

    def ReInitPopulation(self):
        #this functions creates a new population
        self.Population=np.zeros((self.PopulationSize,self.ProblemSize))
        self.Weights=np.zeros(self.PopulationSize)
        self.MinWeights=np.zeros(self.NumIteration)
        for i in range(self.PopulationSize):
            self.Population[i,:]=np.random.permutation(self.ProblemSize)

    def Fittnes(self,Vector):
        #this funciton calculates the total distance traveled between cities
        Distance=0
        LocusWeights=np.zeros(self.ProblemSize)
        for i in range(self.ProblemSize-1):
            Distance += self.DistanceMatrix[int(Vector[i]),int(Vector[i+1])]
            LocusWeights[i+1]=((self.DistanceMatrix[int(Vector[i]),int(Vector[i+1])]
                                -self.MinDistanceFromCity[int(Vector[i+1])])/self.NormalizationFactor[int(Vector[i+1])])+0.1  
        #the first city mutation factor is the average weight   
        return Distance, LocusWeights


    def RandomMutation(self,Vector):
        #select two points in the vector randomly and swap them
        Indices=np.random.choice(self.ProblemSize, 2, replace=False)
        tmp=Vector[Indices[0]]
        Vector[Indices[0]]=Vector[Indices[1]]
        Vector[Indices[1]]=tmp
        return Vector

    def LocusMutation(self,Vector,LocusWeights):
        if np.sum(LocusWeights < 0) > 0:
            print(LocusWeights)
            assert 1 == 0

        else:
            LocusWeights /= LocusWeights.sum()
            #select first index
            Indices=np.random.choice(self.ProblemSize, 2, replace=False,p=LocusWeights)
            #swap them          
            tmp=Vector[Indices[0]]
            Vector[Indices[0]]=Vector[Indices[1]]
            Vector[Indices[1]]=tmp
        return Vector

    def Crossover(self,Parent1, Parent2):
        Offspring1=np.zeros(self.ProblemSize)
        Offspring2=np.zeros(self.ProblemSize)
        SplitIndices = sorted(np.random.choice(self.ProblemSize, 2, replace=False))
        Offspring1[SplitIndices[0]:SplitIndices[1]]=Parent1[SplitIndices[0]:SplitIndices[1]]
        Rest=[]
        Used=set(Parent1[SplitIndices[0]:SplitIndices[1]])
        for i in range(self.ProblemSize):
            if Parent2[i] not in Used:
                Rest.append(Parent2[i])
        Offspring1[:SplitIndices[0]]=Rest[:SplitIndices[0]]
        Offspring1[SplitIndices[1]:]=Rest[SplitIndices[0]:]

        Offspring2[SplitIndices[0]:SplitIndices[1]]=Parent2[SplitIndices[0]:SplitIndices[1]]
        Rest=[]
        Used=set(Parent2[SplitIndices[0]:SplitIndices[1]])
        for i in range(self.ProblemSize):
            if Parent1[i] not in Used:
                Rest.append(Parent1[i])
        Offspring2[:SplitIndices[0]]=Rest[:SplitIndices[0]]
        Offspring2[SplitIndices[1]:]=Rest[SplitIndices[0]:]
        return Offspring1, Offspring2

    def OrderWeights(self,St):
        MinWeight=np.inf
        Weights=np.zeros(self.PopulationSize)
        LocusWeights=np.zeros((self.PopulationSize,self.ProblemSize))
        for i in range(self.PopulationSize):
            Weights[i],LocusWeights[i] = self.Fittnes( self.Population[i,:] )
            if Weights[i]<MinWeight:
                MinWeight=Weights[i]
        self.MinWeights[St]=MinWeight
        #order the Population
        Indices=range(self.PopulationSize)
        Weights, Indices = zip(*sorted(zip(Weights, Indices)))
        Weights=np.asarray(Weights)
        return Indices, Weights, LocusWeights

    def LocusNormalCorssoverMutationSolver(self):
        self.ReInitPopulation() 
        for St in range(self.NumIteration):
            #calc weights
            Indices, Weights,LocusWeights =self.OrderWeights(St)
            
            KeptIndices=Indices[0:int(self.KeepRatio*self.PopulationSize)]
            NewPopulation=np.zeros((self.PopulationSize,self.ProblemSize), dtype=np.int)
            #elitism - kepp the first few as they are
            for a in range(len(KeptIndices)):
                NewPopulation[a,:]=self.Population[KeptIndices[a],:]
            #crossover
            SumWeights= np.sum(Weights)
            for a in range(len(KeptIndices),len(KeptIndices)+int(self.CrossoveredRatio*self.PopulationSize),2):
                Parent1=-1
                Parent2=-1
                #select two instances for crossover
                #crossover for points order
                while (Parent1<0 and Parent2<0) or (Parent1==Parent2):
                    SumP1 = 0
                    Parent1 = 0
                    SelectedSum = np.random.uniform()*SumWeights
                    while SumP1 <= SelectedSum: 
                        SumP1+= Weights[Parent1]
                        Parent1+=1
                    SumP2 = 0
                    Parent2 = 0
                    SelectedSum = np.random.uniform()*SumWeights
                    while SumP2 <= SelectedSum:
                        SumP2+= Weights[Parent2]
                        Parent2+=1
                Parent1=min(Parent1,self.PopulationSize-1)
                Parent2=min(Parent2,self.PopulationSize-1)
                #Generate two offsprings
                Offspring1, Offspring2 = self.Crossover(self.Population[Indices[Parent1],:], self.Population[Indices[Parent2],:])
                NewPopulation[a,:] = Offspring1
                NewPopulation[a+1,:] = Offspring2
            #rest is new
            for a in range(int((1.0-self.NewRatio)*self.PopulationSize),self.PopulationSize):
                NewPopulation[a,:]=np.random.permutation(self.ProblemSize)

            #recalculate weights
            for i in range(self.PopulationSize):
                Weights[i], LocusWeights[i]=self.Fittnes( NewPopulation[i,:] )      
            #random mutation -switch
            for a in range(2,self.PopulationSize):
                while np.random.uniform()<self.MutationRate:
                    NewPopulation[a,:]=self.LocusMutation(NewPopulation[a,:],LocusWeights[a,:]) 
                    _, LocusWeights[a,:] = self.Fittnes(NewPopulation[a,:])     
            self.Population=NewPopulation
        return self.MinWeights

    def BaselineSolver(self):
        self.ReInitPopulation() 
        for St in range(self.NumIteration):
            #calc weights
            Indices, Weights,LocusWeights =self.OrderWeights(St)
            
            KeptIndices=Indices[0:int(self.KeepRatio*self.PopulationSize)]
            NewPopulation=np.zeros((self.PopulationSize,self.ProblemSize), dtype=np.int)
            #elitism - kepp the first few as they are
            for a in range(len(KeptIndices)):
                NewPopulation[a,:]=self.Population[KeptIndices[a],:]
            #crossover
            SumWeights= np.sum(Weights)
            for a in range(len(KeptIndices),len(KeptIndices)+int(self.CrossoveredRatio*self.PopulationSize),2):
                Parent1=-1
                Parent2=-1
                #select two instances for crossover
                #crossover for points order
                while (Parent1<0 and Parent2<0) or (Parent1==Parent2):
                    SumP1 = 0
                    Parent1 = 0
                    SelectedSum = np.random.uniform()*SumWeights
                    while SumP1 <= SelectedSum:
                        SumP1+= Weights[Parent1]
                        Parent1+=1
                    SumP2 = 0
                    Parent2 = 0
                    SelectedSum = np.random.uniform()*SumWeights
                    while SumP2 <= SelectedSum:
                        SumP2+= Weights[Parent2]
                        Parent2+=1
                Parent1=min(Parent1,self.PopulationSize-1)
                Parent2=min(Parent2,self.PopulationSize-1)
                #Generate two offsprings
                Offspring1, Offspring2 = self.Crossover(self.Population[Indices[Parent1],:], self.Population[Indices[Parent2],:])
                NewPopulation[a,:] = Offspring1
                NewPopulation[a+1,:] = Offspring2
            #rest is new
            for a in range(int((1.0-self.NewRatio)*self.PopulationSize),self.PopulationSize):
                NewPopulation[a,:]=np.random.permutation(self.ProblemSize)      
            #random mutation -switch
            for a in range(2,self.PopulationSize):
                while np.random.uniform()<self.MutationRate:
                    NewPopulation[a,:]=self.RandomMutation(NewPopulation[a,:])          
            self.Population=NewPopulation
        return self.MinWeights    
    
    
class SATest():
    def __init__(self,tsp,Tries=10,Iteration=50,Searches=10):
        #problem parameters
        self.ProblemSize=len(tsp) #number of cities
        self.NumTries=Tries
        self.NumSearches=Searches
        self.NumIteration=Iteration
        self.tsp = tsp
        self.DistanceMatrix=np.zeros((self.ProblemSize,self.ProblemSize))
        self.InitialMutationRate=0.8
        self.MutationDecrease=0.8/float(self.NumIteration) # at the last iteration mutation rate becomes zero
        self.GenerateProblem()


        
    def GenerateProblem(self):
        OrigDistanceMatrix=np.zeros((self.ProblemSize,self.ProblemSize))
        for i in range(len(tsp)):
            for j in range(i+1,len(tsp)):
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
        #this funciton calculates the total distance traveled between cities
        Distance=0
        LocusWeights=np.zeros(self.ProblemSize)
        for i in range(self.ProblemSize-1):
            Distance += self.DistanceMatrix[int(Vector[i]),int(Vector[i+1])]
            LocusWeights[i+1]=((self.DistanceMatrix[int(Vector[i]),int(Vector[i+1])]
                                -self.MinDistanceFromCity[int(Vector[i+1])])/self.NormalizationFactor[int(Vector[i+1])])+0.1  
        #the first city mutation factor is the average weight   
        return Distance, LocusWeights


    def RandomMutation(self,Vector):
        #select two points in the vector randomly and swap them
        Indices=np.random.choice(self.ProblemSize, 2, replace=False)
        tmp=Vector[Indices[0]]
        Vector[Indices[0]]=Vector[Indices[1]]
        Vector[Indices[1]]=tmp
        return Vector

    def LocusMutation(self,Vector,LocusWeights):
        if np.sum(LocusWeights < 0) > 0:
            print(LocusWeights)
            assert 1 == 0

        else:
            LocusWeights /= LocusWeights.sum()
            #select first index
            Indices=np.random.choice(self.ProblemSize, 2, replace=False,p=LocusWeights)
            #swap them          
            tmp=Vector[Indices[0]]
            Vector[Indices[0]]=Vector[Indices[1]]
            Vector[Indices[1]]=tmp
        return Vector

    def LocusSA(self):
        Solution=np.random.permutation(self.ProblemSize)
        Weight, LWeight=self.Fittnes( Solution)
        BestWeight=Weight
        BestSolution=np.copy(Solution)
        for Try in range(self.NumTries):
            self.MutationRate=self.InitialMutationRate
            #generate random solution
            Solution=np.random.permutation(self.ProblemSize) 
            OrigFitness,OrigLocus=  self.Fittnes( Solution)
            for It in range(self.NumIteration):
                  for Search in range(self.NumIteration):
                        for i in range(self.NumSearches):
                              SolutionCandidate=np.copy(Solution)
                              while np.random.uniform()<self.MutationRate:
                                    CandFitness,CandLocus=  self.Fittnes( SolutionCandidate)
                                    SolutionCandidate=self.LocusMutation(SolutionCandidate,CandLocus) 
                        CandidateFitness,_=  self.Fittnes( SolutionCandidate)
                        if (CandidateFitness<OrigFitness):
                              Solution=SolutionCandidate
                              OrigFitness=CandidateFitness
                  if OrigFitness<BestWeight:
                        BestWeight=OrigFitness
                        BestSolution=np.copy(Solution)
        
                  self.MutationRate-= self.MutationDecrease         
        return BestWeight 
        
    def BaselineSA(self):
        Solution=np.random.permutation(self.ProblemSize)
        Weight, LWeight=self.Fittnes( Solution)
        BestWeight=Weight
        BestSolution=np.copy(Solution)
        for Try in range(self.NumTries):
            self.MutationRate=self.InitialMutationRate
            #generate random solution
            Solution=np.random.permutation(self.ProblemSize) 
            OrigFitness,_=  self.Fittnes( Solution)
            for It in range(self.NumIteration):
                  for Search in range(self.NumIteration):
                        for i in range(self.NumSearches):
                              SolutionCandidate=np.copy(Solution)
                              while np.random.uniform()<self.MutationRate:
                                    SolutionCandidate=self.RandomMutation(SolutionCandidate) 
                        CandidateFitness,_=  self.Fittnes( SolutionCandidate)
                        if (CandidateFitness<OrigFitness):
                              Solution=SolutionCandidate
                              OrigFitness=CandidateFitness
                  if OrigFitness<BestWeight:
                        BestWeight=OrigFitness
                        BestSolution=np.copy(Solution)
        
                  self.MutationRate-= self.MutationDecrease         
        return BestWeight 
  
          
DataDir='./routs'
Problems= ['att48.tsp', 'berlin52.tsp', 'burma14.tsp', 'eil51.tsp', 'eil76.tsp', 'pr76.tsp', 'rat99.tsp', 'st70.tsp', 'ulysses16.tsp', 'ulysses22.tsp','kroA100.tsp','kroB100.tsp','kroC100.tsp','kroD100.tsp','kroE100.tsp','rd100.tsp']


with open('results.csv', 'w', newline='') as f:
      writer = csv.writer(f)
      for Problem in Problems:
          print(Problem)
          writer.writerow([Problem])
          for tries in range(10):
                print(tries)
                tsp = read_data('./routs/' + Problem)
                GA=GATest(tsp,S=len(tsp),P=100,I=20)
                SA=SATest(tsp,Tries=100,Iteration=20,Searches=5)
                
                GaLargeNormalResult=GA.LocusNormalCorssoverMutationSolver()[-1]
                GaLargeLocusResult=GA.BaselineSolver()[-1]
                SaLargeNormalResult=SA.BaselineSA()
                SaLargeLocusResult=SA.LocusSA()

                GA=GATest(tsp,S=len(tsp),P=10,I=20)
                SA=SATest(tsp,Tries=10,Iteration=20,Searches=2)
                
                GaSmallNormalResult=GA.LocusNormalCorssoverMutationSolver()[-1]
                GaSmallLocusResult=GA.BaselineSolver()[-1]
                SaSmallNormalResult=SA.BaselineSA()
                SaSmallLocusResult=SA.LocusSA()

                writer.writerow([GaLargeLocusResult, GaLargeNormalResult, SaLargeNormalResult, SaLargeLocusResult, GaSmallLocusResult, GaSmallNormalResult,  SaSmallNormalResult, SaSmallLocusResult])