import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy as np
import copy


#%% INPUT DATA

ts = np.loadtxt("trainingSet.txt", 
                usecols=[1,2,3])
#np.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])

# X = np.arange(-10,10,0.1)

# X = ts[:,0:1]
X = ts[:,0:2]

Y = ts[:,2]

#%% MEMBERSHIP FUNCTIONS

mf1 = [
        ['gaussmf',{'mean':-5.,'sigma':1.}],
        ['gbellmf',{'a':1.,'b':1.,'c':1.}],
        # ['sigmf',{'b':5.,'c':0.5}]
        # ['gaussmf',{'mean':5.,'sigma':1.}],
       ]

mf2 = [
       ['gaussmf',{'mean':1.,'sigma':2.}],
        ['gaussmf',{'mean':2.,'sigma':3.}],
       # ['gaussmf',{'mean':-2.,'sigma':10.}],
       # ['gaussmf',{'mean':-10.5,'sigma':5.}]
       ]

mf = [
       mf1, 
       mf2
      ]

mfc = membership.membershipfunction.MemFuncs(mf)
print(mfc)

#%% ANFIS Object

anf = anfis.ANFIS(X, Y, mfc)
print(anf)

#%% MF Plots

print("Plotting mf")
anf.plotMF(0)
anf.plotMF(1)

#%% Training

anf.trainHybridJangOffLine(epochs=10)
print(round(anf.consequents[-1][0],6)) 
print(round(anf.consequents[-2][0],6)) 
print(round(anf.fittedValues[9][0],6))
if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
	print('test is good')

print("Plotting errors")
#anf.plotErrors()
print("Plotting results")
# anf.plotResults()

#%% MF Plots

print("Plotting mf")
anf.plotMF(0)
anf.plotMF(1)

