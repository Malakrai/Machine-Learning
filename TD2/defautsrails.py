import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import svm


# charger les données de défauts de rails
data = np.loadtxt("defautsrails.dat")
X = data[:,:-1]  # tout sauf dernière colonne
y = data[:,-1]   # uniquement dernière colonne

