import numpy as np
import pickle
from tensornetworks.PositiveMPS import PositiveMPS
from tensornetworks.RealBorn import RealBorn
from tensornetworks.ComplexBorn import ComplexBorn
from tensornetworks.RealLPS import RealLPS
from tensornetworks.ComplexLPS import ComplexLPS
from tensornetworks.ComplexCircularLPS import ComplexCircularLPS

with open('datasets/lymphography', 'rb') as f:
	a=pickle.load(f)
X=a[0]
X=X.astype(int)
print("--MPS--")
mps = PositiveMPS(D=3, learning_rate=0.2, batch_size=20, n_iter=30, verbose=True)
mps.fit(X)
print(mps.likelihood(X))

print("\n--RealLPS--")
mps3 =  RealLPS(D=3, learning_rate=0.2, batch_size=20, n_iter=30, verbose=True)
mps3.fit(X)
print("\n--ComplexLPS--")
print(mps3.likelihood(X))
mps4 =  ComplexLPS(D=3, learning_rate=0.2, batch_size=20, n_iter=30, verbose=True)
mps4.fit(X)
print(mps4.likelihood(X))

print("\n--ComplexCircularLPS--")
mps5 =  ComplexCircularLPS(D=3, learning_rate=0.2, batch_size=20, n_iter=30, verbose=True)


mps5.fit(X,w_init=mps4.w)
print(mps5.likelihood(X))
