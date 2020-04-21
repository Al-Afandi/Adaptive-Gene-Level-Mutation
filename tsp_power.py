
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
import itertools as it


# In[3]:


class GATest():
    def __init__(self,S=128,P=400,I=30):
        #problem parameters
        self.ProblemSize=S #number of cities
        self.DistanceMatrix=np.zeros((self.ProblemSize,self.ProblemSize))
        self.PopulationSize=P
        self.NumIteration=I

        #parameters of the algorithm
        self.KeepRatio=0.5 #this proportion of the population will be kept
        self.NewRatio=0.2 #this proportion iwll be newly generated
        self.CrossoveredRatio=1-(self.KeepRatio+self.NewRatio) #this proprtion will be created as new offsprings
        self.MutationRate=0.3 #probability of mutation of an instance
        self.Population=np.zeros((self.PopulationSize,self.ProblemSize)) #population is stored in this table
        self.Weights=np.zeros(self.PopulationSize) 
        self.MinWeights=np.zeros(self.NumIteration)
        self.GenerateProblem()
        self.ReInitPopulation()
    
    def setter(self,KeepRatio,NewRatio,MutationRate):
        self.KeepRatio=KeepRatio #this proportion of the population will be kept
        self.NewRatio=NewRatio #this proportion iwll be newly generated
        self.MutationRate=MutationRate #probability of mutation of an instance

    def GenerateProblem(self):
        OrigDistanceMatrix=np.zeros((self.ProblemSize,self.ProblemSize))
        RandomPoints = np.random.random_sample((self.ProblemSize,2))
        for i in range(len(RandomPoints)):
            for j in range(i+1,len(RandomPoints)):
                OrigDistanceMatrix[i,j] = np.sqrt((RandomPoints[i][0]-RandomPoints[j][0])**2+(RandomPoints[i][1]-RandomPoints[j][1])**2)
                OrigDistanceMatrix[j,i] = OrigDistanceMatrix[i,j] + 0.05
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
                                -self.MinDistanceFromCity[i])
                               /self.NormalizationFactor[i])+1
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
        LocusWeights[Firstind]=np.max(LocusWeights)
        LocusWeights = np.max(LocusWeights) - LocusWeights
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
    
    def CrossLocusNormalCorssoverMutationSolver(self,Power):
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
                    NewPopulation[a,:]=self.LocusMutation(NewPopulation[a,:],LocusWeights[a,:]**Power)
                    _, TempWeight = self.Fittnes(NewPopulation[a,:]) 
                    if(np.sum(TempWeight) == np.sum(LocusWeights[a,:])):
                        NewPopulation[a,:]=self.CrossLocusMutation(NewPopulation[a,:],LocusWeights[a,:])
                        _, TempWeight = self.Fittnes(NewPopulation[a,:]) 
                    LocusWeights[a,:] = TempWeight       
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
                    _, TempWeight = self.Fittnes(NewPopulation[a,:]) 
                    if(np.sum(TempWeight) == np.sum(LocusWeights[a,:])):
                        NewPopulation[a,:]=self.CrossLocusMutation(NewPopulation[a,:],LocusWeights[a,:])
                        _, TempWeight = self.Fittnes(NewPopulation[a,:]) 
                    LocusWeights[a,:] = TempWeight       
            self.Population=NewPopulation
        return self.MinWeights

    def BaselineSolver(self,P):
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
            #random mutation -switch
            for a in range(2,self.PopulationSize):
                while np.random.uniform()<self.MutationRate:
                    NewPopulation[a,:]=self.RandomMutation(NewPopulation[a,:]**P)          
            self.Population=NewPopulation
        return self.MinWeights

    def JustRandomMutationSolver(self):
        #self.ReInitPopulation() 
        for St in range(self.NumIteration):
            #calc weights
            Indices, Weights,LocusWeights =self.OrderWeights(St)
            
            KeptIndices=Indices[0:int(self.KeepRatio*self.PopulationSize)]
            NewPopulation=np.zeros((self.PopulationSize,self.ProblemSize), dtype=np.int)
            #elitism - kepp the first few as they are
            for a in range(len(KeptIndices)):
                NewPopulation[a,:]=self.Population[KeptIndices[a],:]
            #rest is new
            for a in range(len(KeptIndices),self.PopulationSize):
                NewPopulation[a,:]=np.random.permutation(self.ProblemSize)      
            #random mutation -switch
            for a in range(2,self.PopulationSize):
                while np.random.uniform()<self.MutationRate:
                    NewPopulation[a,:]=self.RandomMutation(NewPopulation[a,:])          
            self.Population=NewPopulation
        return self.MinWeights


# In[4]:


GA=GATest()
Population = GA.Population 
repeat=10
Baseline=np.zeros((repeat,30))
Locus=np.zeros((repeat,30))
Powers=[]
for i in range(0,8):
    Powers.append(i)

PowersValue = np.zeros((len(Powers)+1,30))
counter=0
for i in Powers:
    for r in range(0,repeat):
        LocusMinWeights=GA.LocusNormalCorssoverMutationSolver(10**i)
        Locus[r,:]   =LocusMinWeights
        GA.Population = Population 
    PowersValue[counter]= np.mean(Locus,0)
    print(counter)
    counter+=1  
    
for r in range(0,repeat):
    LocusMinWeights=GA.LInfLocusNormalCorssoverMutationSolver()
    Locus[r,:]   =LocusMinWeights
    GA.Population = Population 
    
PowersValue[counter]= np.mean(Locus,0)


# In[5]:


plt.figure()
plt.plot(np.mean(PowersValue,1),'b')
plt.savefig('./tsp_diffrent_power/TableSize_124_Genom_400_Mutation_0.3_power_0_8_Inf.png')  
plt.show()


# In[7]:


plt.plot(np.min(PowersValue[:-1],1),'b')
plt.show()



