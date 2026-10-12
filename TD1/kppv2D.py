import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import neighbors



#### programme principal à implémenter dans cette fonction ####
def monprogramme(Xapp, Yapp, K):
	"""
		Xapp, Yapp : base d'apprentissage générée avec la souris
		K	: paramètre réglé par +/-
	"""
	print("Apprentissage lancé avec " + str(len(Xapp)) + " points et K = ", K)

	# à compléter pour apprendre le modèle KPPV...
	kppv = neighbors.KNeighborsClassifier(K)
	#apprentissage sur X,Y 
	kppv.fit(Xapp, Yapp)
    

	# création d'une grille de points de test
	r1 = np.arange(-5,5,0.2)
	Xtest = np.zeros((len(r1)*len(r1),2))
	i = 0
	for x1 in r1:
		for x2 in r1:
			Xtest[i,:] = [x1, x2]
			i += 1
	
	# Prédire la catégorie pour tous les points de test...
	Ypred = kppv.predict(Xtest)
	Ypred_app = kppv.predict(Xapp)
	# ... et tracer le résultat avec par exemple 
	Xbleu = Xtest[ Ypred ==1,:]
	Xrouge = Xtest[ Ypred ==2,:] # dans notre modele le -1 est un  2
	
    

	# plot(.... 'ob', alpha=0.2) pour des gros points bleus semi-transparents
	plt.plot(Xbleu[:,0], Xbleu[:,1], 'ob', alpha=0.2) #pour des gros points bleus semi-transparents on fait alpha = 0.2
	plt.plot(Xrouge[:,0], Xrouge[:,1], 'or', alpha=0.2)

	#Erreur d'apprentissage
	
	Remp =np.mean(Ypred_app != Yapp)
	
	print("Erreur d'apprentissage = " , Remp)
    

	# pour réellement mettre à jour le graphique: 
	fig.canvas.draw()

	


	



##### Gestion de l'interface graphique ########


Xplot = np.zeros((0,2))
Yplot = np.zeros(0)
plotvariance = 0

# lorsque K augmente l'erreur d'app augmente alors que la complexite diminue si K augmente 
K = 5 

def onclick(event):
	global Xplot
	global Yplot
	
	
	if plotvariance == 0:
		newX = np.array([[event.xdata,event.ydata]])
	else:
		newX = math.sqrt(plotvariance) * np.random.randn(10, 2) + np.ones((10,1)).dot(np.array([[event.xdata,event.ydata]]))

	print("Ajout de " + str(len(newX)) + " points en (" + str(event.xdata) + ", " + str(event.ydata) + ")")

	Xplot = np.concatenate((Xplot,newX))
	if event.button == 1 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.b')
		newY = np.ones(len(newX)) * 1
	elif event.button == 3 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.r')
		newY = np.ones(len(newX)) * 2
	elif event.button == 1 and event.key == "shift":
		plt.plot(newX[:,0], newX[:,1],'.g')
		newY = np.ones(len(newX)) * 3
	elif event.button == 3 and event.key == "shift":
		plt.plot(newX[:,0], newX[:,1],'.m')
		newY = np.ones(len(newX)) * 4
	Yplot = np.concatenate((Yplot,newY))
	
	fig.canvas.draw()


def onscroll(event):
	global plotvariance
	if event.button == "up":
		plotvariance = round(plotvariance + 0.2, 1)
	elif event.button == "down" and plotvariance > 0.1:
		plotvariance = round(plotvariance - 0.2, 1)
	print("Variance = ", plotvariance)

def onkeypress(event):
	global plotvariance
	global K
	if event.key == " ":
		monprogramme(Xplot, Yplot, K)
	elif event.key == "+" and K < len(Xplot):
		K += 1
		print("K = " , K)
	elif event.key == "-" and K > 1:
		K -= 1
		print("K = " , K)
	elif event.key == "V":
		plotvariance = round(plotvariance + 0.2, 1)
		print("Variance = ", plotvariance)
	elif event.key == "v" and plotvariance > 0.1:
		plotvariance = round(plotvariance - 0.2, 1)
		print("Variance = ", plotvariance)
		
	
fig = plt.figure()

plt.axis([-5, 5, -5, 5])

cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("scroll_event", onscroll)
cid3 = fig.canvas.mpl_connect("key_press_event", onkeypress)

print("Utilisez la souris pour ajouter des points à la base d'apprentissage :")
print(" clic gauche : points bleus")
print(" clic droit : points bleus")
print(" MAJ + clic gauche : points verts")
print(" MAJ + clic droit : points magenta")
print("\nMolette : +/- variance ")
print("   si variance = 0  => ajout d'un point")
print("   si variance > 0  => ajout de points selon une loi gaussienne")
print("\n ESPACE pour lancer la fonction monprogramme(Xapp,Yapp,K)")
print("    avec la valeur de K modifiée par +/-\n\n") 

plt.show()
