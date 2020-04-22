import matplotlib
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools as it
import os
import pickle

class GA():
    def __init__(self,S=128,P=400,I=20):
        #problem parameters
        self.ProblemSize=S #problem size
        self.PopulationSize=P
        self.NumIteration=I

        #parameters of the algorithm
        self.KeepRatio=0.6 #this proportion of the population will be kept
        self.NewRatio=0.2 #this proportion iwll be newly generated
        self.CrossoveredRatio=1-(self.KeepRatio+self.NewRatio) #this proprtion will be created as new offsprings
        self.MutationRate=0.3 #probability of mutation of an instance
        self.Population=np.zeros((self.PopulationSize,self.ProblemSize)) #population is stored in this table
        self.Weights=np.zeros(self.PopulationSize) 
        self.MinWeights=np.zeros(self.NumIteration)
        self.ReInitPopulation()

    def setter(self,KeepRatio,NewRatio,MutationRate):
        self.KeepRatio=KeepRatio #this proportion of the population will be kept
        self.NewRatio=NewRatio #this proportion iwll be newly generated
        self.MutationRate=MutationRate #probability of mutation of an instance
        self.CrossoveredRatio=1-(self.KeepRatio+self.NewRatio)
        
    def ReInitPopulation(self):
        #this functions creates a new population
        self.Population=np.zeros((self.PopulationSize,self.ProblemSize))
        self.Weights=np.zeros(self.PopulationSize)
        self.MinWeights=np.zeros(self.NumIteration)
        for i in range(self.PopulationSize):
            self.Population[i,:]=np.random.permutation(self.ProblemSize)

    def Fittnes(self,Vector):
        #this funciton calculates the number of hitting queens in a vector
        NumHits=0
        LocusWeights=np.ones(self.ProblemSize)*0.1
        for i in range(self.ProblemSize):
            for j in range(i+1,self.ProblemSize):
                #queens can only hit each other in the diagonal because of the problem represenatation, so that is what we have to check
                if (Vector[i]==Vector[j]) or (Vector[i]+i==Vector[j]+j) or (((self.ProblemSize-1)-Vector[i])+((self.ProblemSize-1)-i)==((self.ProblemSize-1)-Vector[j])+((self.ProblemSize-1)-j)):
                    NumHits+=1
                    LocusWeights[i]+=1
                    LocusWeights[j]+=1
        return NumHits, LocusWeights


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
    
    def CrossLocusMutation(self,Vector,LocusWeights):
        #select first index
        Sum=0
        S=np.sum(LocusWeights)
        Limit=np.random.uniform()*S
        ind=0
        while Sum<Limit:
            Sum+=LocusWeights[ind]
            ind+=1
        Firstind=ind-1
        LocusWeights[Firstind] = np.max(LocusWeights)
        #select second index
        LocusWeights = np.max(LocusWeights) - LocusWeights
        Sum=0
        S=np.sum(LocusWeights)
        Limit=np.random.uniform()*S
        ind=0
        while Sum<Limit:
            Sum+=LocusWeights[ind]
            ind+=1
        Secondind=ind-1 
        #swap them          
        tmp=Vector[Firstind]
        Vector[Firstind]=Vector[Secondind]
        Vector[Secondind]=tmp
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

    def LocusNormalCorssoverMutationSolver(self,power):
        #self.ReInitPopulation() 
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
                    NewPopulation[a,:]=self.LocusMutation(NewPopulation[a,:],LocusWeights[a,:]**power) 
                    _, LocusWeights[a,:] = self.Fittnes(NewPopulation[a,:])     
            self.Population=NewPopulation
        return self.MinWeights

    
    def LInfLocusNormalCorssoverMutationSolver(self):
        #self.ReInitPopulation() 
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
                Weights[i], LocusWeight=self.Fittnes( NewPopulation[i,:] )
                ind = np.argmax(LocusWeight)
                LInf = LocusWeight[ind]
                LocusWeight = np.ones_like(LocusWeight)*0.1
                LocusWeight[ind] = LInf
                LocusWeights[i] = LocusWeight
                
            
            #random mutation -switch
            for a in range(2,self.PopulationSize):
                while np.random.uniform()<self.MutationRate:
                    NewPopulation[a,:]=self.LocusMutation(NewPopulation[a,:],LocusWeights[a,:])
                    _, LocusWeights[a,:] = self.Fittnes(NewPopulation[a,:])         
            self.Population=NewPopulation
        return self.MinWeights
    

TableSize=64
NumberOfGerneration = 20
GenomeSize=200
KeepRatio=0.05
MutationRate=0.3
repeat=20
Locus=np.zeros((repeat,NumberOfGerneration ))
Powers=[]
for i in range(0,8):
    Powers.append(i)    
PowersValue = np.zeros((len(Powers)+1,repeat))
Data = np.zeros((len(Powers)+1,repeat,NumberOfGerneration))
counter=0


GAI=GA() #genetic algorithm instance
Population=GAT.Population

for i in Powers:
    for r in range(0,repeat):
        LocusMinWeights=GAI.LocusNormalCorssoverMutationSolver(10**i)
        Locus[r,:]   =LocusMinWeights
        Data[i,r,:]= LocusMinWeights
        GAI.Population = Population  
    PowersValue[counter]= np.min(Locus,1)
    print(counter)
    counter+=1  
    
for r in range(0,repeat):
    LocusMinWeights=GAI.LInfLocusNormalCorssoverMutationSolver()
    Locus[r,:]   =LocusMinWeights
    Data[len(Powers),r,:]= LocusMinWeights
    GAI.Population = Population 
    
PowersValue[counter]= np.min(Locus,1)

'''
The fitness value of the best optimal solution for 64-Queens problem as a function of Powers where we averaged out ten runs.
It starts with uniform distribution then uses a logarithmic scale of Pow, and end up with L-infinite norm. 
'''

plt.figure()
plt.plot(np.mean(PowersValue,1),'b')
plt.show()
