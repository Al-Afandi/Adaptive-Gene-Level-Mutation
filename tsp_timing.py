import pickle
import random
import numpy as np
import itertools
import matplotlib.pyplot as plt


#exahustive search of the original TSP
def GetSolutionBruteForce(CostMatrix, AllSolutions):
    MinCost=float('Inf')
    BestSolution=[]
    for ind in range(AllSolutions.shape[0]):
        CurrentCost=0
        for ind2 in range(AllSolutions.shape[1]-1):
            From=AllSolutions[ind][ind2]
            To=AllSolutions[ind][ind2+1]
            CurrentCost+=CostMatrix[From][To]
        if CurrentCost<MinCost:
            MinCost=CurrentCost
            BestSolution=AllSolutions[ind]
    return BestSolution, MinCost

class GATest():
    def __init__(self,OrigDistanceMatrix,S=32,P=200,I=50):
        #problem parameters
        self.ProblemSize=S #number of cities
        self.PopulationSize=P
        self.NumIteration=I
        self.DistanceMatrix=np.zeros((self.ProblemSize,self.ProblemSize))

        #parameters of the algorithm
        self.KeepRatio=0.5 #this proportion of the population will be kept
        self.NewRatio=0.2 #this proportion iwll be newly generated
        self.CrossoveredRatio=1-(self.KeepRatio+self.NewRatio) #this proprtion will be created as new offsprings
        self.MutationRate=0.3 #probability of mutation of an instance
        self.Population=np.zeros((self.PopulationSize,self.ProblemSize)) #population is stored in this table
        self.Weights=np.zeros(self.PopulationSize) 
        self.MinWeights=np.zeros(self.NumIteration)
        self.GenerateProblem(OrigDistanceMatrix)
        self.ReInitPopulation()
    
    def setter(self,KeepRatio,NewRatio,MutationRate):
        self.KeepRatio=KeepRatio #this proportion of the population will be kept
        self.NewRatio=NewRatio #this proportion iwll be newly generated
        self.MutationRate=MutationRate #probability of mutation of an instance
        self.CrossoveredRatio=1-(self.KeepRatio+self.NewRatio) #this proprtion will be created as new offsprings

    def GenerateProblem(self,OrigDistanceMatrix):
        #OrigDistanceMatrix=np.random.normal(100,30,(self.ProblemSize,self.ProblemSize))
        #put infinite values into the diagonal
        self.DistanceMatrix= np.copy(OrigDistanceMatrix)
        self.DistanceMatrix+= np.where(np.eye(OrigDistanceMatrix.shape[0])>0,-1*np.inf,0)
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
            LocusWeights[i+1]=((self.DistanceMatrix[int(Vector[i]),int(Vector[i+1])]-self.MinDistanceFromCity[i])/self.NormalizationFactor[i])+0.1  
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
        #select first index
        Sum=0
        S=np.sum(LocusWeights)
        Limit=np.random.uniform()*S
        ind=0
        while Sum<Limit:
            Sum+=LocusWeights[ind]
            ind+=1
        Firstind=ind-1
        LocusWeights[Firstind]=0
        #select second index
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


ProblemSize=10
DistanceMatrix=np.zeros((ProblemSize,ProblemSize))
RandomPoints = np.random.random_sample((ProblemSize,2))
for i in range(len(RandomPoints)):
    for j in range(i+1,len(RandomPoints)):
        DistanceMatrix[i,j] = np.sqrt((RandomPoints[i][0]-RandomPoints[j][0])**2+(RandomPoints[i][1]-RandomPoints[j][1])**2)
        DistanceMatrix[j,i] = DistanceMatrix[i,j] + 0.05

AllSolutions=np.asarray(list(itertools.permutations(range(ProblemSize))))
MinDistanceCost=float('Inf')

"""
This Figure depicts TSP Problem with 10 cities manifesting the speed and theability of our approach to nearly reach the optimal solution
in comparison with the traditional approach. All the experiments have been conducted with 200 populationsize, 200 generations and 
0.5 mutation rate. The optimal solution has been obtained using brute force algorithm. The experiments were repeated 100 times.
"""

NumGeneration=200
PopulationSize=100
repeat=100
RandomMinWeights = np.zeros((repeat,NumGeneration))
LocusMinWeights  = np.zeros((repeat,NumGeneration))
for i in range(repeat):
    GA=GATest(DistanceMatrix, ProblemSize,PopulationSize,NumGeneration)
    RandomMinWeights[i,:] = GA .BaselineSolver()
    LocusMinWeights[i,:]  = GA.LocusNormalCorssoverMutationSolver()
solution = GetSolutionBruteForce(DistanceMatrix,AllSolutions)


plt.figure(figsize=(12, 7))
plt.plot(np.mean(RandomMinWeights,0),"b",np.mean(LocusMinWeights,0),"r",np.ones(len(LocusMinWeights[0]))*solution[1])
plt.legend(["Normal", "Locus",'Optimal'], loc=0,prop={'size': 26})
plt.show()




"""
This Figure depicts TSP Problem with 10 cities manifesting the ability of ourapproach to surpass the traditional approach
consuming the same time. 
We did run the the traditional approach for two times more generations e.g. if axis x equals 25 generations for locus mutation,
it will equal 25*2 generations for traditional mutation.
All the experiments have been conducted with 200 population size and 0.5 mutationrate. 
We run the traditional approach for 600 generations while we run locus mutationonly for 300 generations. 
The optimal solution has been obtained using brute forcealgorithm and the experiments were repeated 100 times.
"""

NumGeneration=50
PopulationSize=200
repeat=100

LocusMinWeights  = np.zeros((repeat,NumGeneration))
for i in range(repeat):
    GA=GATest(DistanceMatrix, ProblemSize,PopulationSize,NumGeneration)
    LocusMinWeights[i,:]  = GA.LocusNormalCorssoverMutationSolver()

RandomMinWeights = np.zeros((repeat,NumGeneration*2))
for i in range(repeat):
    GA=GATest(DistanceMatrix, ProblemSize,PopulationSize,NumGeneration*2)
    RandomMinWeights[i,:] = GA.BaselineSolver()
    
MeanRandom = []
    
MeanRandomMinWeights = np.mean(RandomMinWeights,0)
for i in range(0,len(MeanRandomMinWeights),2):
    MeanRandom.append(MeanRandomMinWeights[i])


plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)

plt.figure(figsize=(12, 7))
plt.plot(MeanRandom,"r",np.mean(LocusMinWeights,0),"b",np.ones(len(LocusMinWeights[0]))*solution[1])
plt.legend(["Normal", "Locus",'Optimal'], loc=0,prop={'size': 26})
plt.show()
