import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn import neighbors
	



#### Fonctions de chargement et affichage de la base mnist ####

def load_mnist(m,mtest):

	X = np.load("mnistX.npy")
	y = np.load("mnisty.npy")

	random_state = check_random_state(0)
	permutation = random_state.permutation(X.shape[0])
	X = X[permutation]
	y = y[permutation]
	X = X.reshape((X.shape[0], -1))

	return train_test_split(X, y, train_size=m, test_size=mtest)


def showimage(x):
	plt.imshow( 255 - np.reshape(x, (28, 28) ), cmap="gray")
	plt.show()
	

#############################
#### Programme principal ####

# chargement de la base mnist:
Xtrain, Xtest, ytrain, ytest = load_mnist(60000, 1000)

print("Taille de la base d'apprentissage : ", Xtrain.shape)

#showimage(Xtrain[0])
erreur = np.zeros(10)
for k in range(1,11):
	kppv = neighbors.KNeighborsClassifier(k)
	kppv.fit(Xtrain, ytrain)
	Ypred =kppv.predict(Xtest)


	erreur[k-1] = np.mean(Ypred != ytest)


	print("Erreur de test = " , erreur[k-1],"pour la valeur de k", k )


meilleur_K = np.argmin(erreur) + 1
print("Meilleur K = ", meilleur_K,'avec une erreur de test de ', erreur[meilleur_K-1]*100)

plt.plot(range(1, 11), erreur, marker="o")
plt.xlabel("Valeur de K")
plt.ylabel("Erreur de test")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()





#on decoupe la base de l'apprentissage en deux parties: une partie pour l'apprentissage et une partie pour la validation

Xvalid=Xtrain[-1000:,:]
yvalid=ytrain[-1000:]
Xapp=Xtrain[:-1000,:]
yapp=ytrain[:-1000]

erreur_valid = np.zeros(10)
for k in range(1,11):
	kppv = neighbors.KNeighborsClassifier(k)
	kppv.fit(Xapp, yapp)
	Ypred =kppv.predict(Xvalid)


	erreur_valid[k-1] = np.mean(Ypred != yvalid)


	print("Erreur de validation = " , erreur_valid[k-1],"pour la valeur de k", k )


meilleur_K = np.argmin(erreur_valid) + 1
print("Meilleur K = ", meilleur_K, 'avec une erreur de validation de ', erreur_valid[meilleur_K-1]*100)

plt.plot(range(1, 11), erreur_valid, marker="o")
plt.xlabel("Valeur de K")
plt.ylabel("Erreur de validation")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

#memoir 60000*(784+1)*8octet = 


# Implémentez vous-même l’algorithme Kppv dans une fonction kppvpredict(Xtest, Xapp,Yapp, K) et essayez de voir comment rendre les calculs plus rapides.


ypred =kppv.predict(Xtest, Xapp, yapp,k)

Distance = np.linalg.norm(Xapp-x, axis=1)


