import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm



#### programme principal à implémenter dans cette fonction ####
def monprogramme(Xapp, Yapp, C):
    """
        Programme pour les SVM linéaires (lancé avec ESPACE)
        
        Xapp, Yapp : base d'apprentissage générée avec la souris
        C	: paramètre réglé par +/-
    """
    print("Apprentissage lancé avec " + str(len(Xapp)) + " points et C = ", C)

    # à compléter pour apprendre le modèle SVM...
    # (svm.LinearSVC refuse C = math.inf dans les versions récentes de sklearn,
    #  on utilise donc SVC avec un noyau linéaire, qui donne le même modèle)
    model = svm.SVC(C=C, kernel='linear')
    model.fit(Xapp, Yapp)

    # création d'une grille de points de test
    r1 = np.arange(-5,5,0.2)
    Xtest = np.zeros((len(r1)*len(r1),2))
    i = 0
    for x1 in r1:
        for x2 in r1:
            Xtest[i,:] = [x1, x2]
            i += 1

    # Prédire la catégorie pour tous les points de test...
    Ypred = model.predict(Xtest)

    # ... et tracer le résultat avec par exemple
    plt.cla()
    plt.plot(Xtest[Ypred == 1, 0], Xtest[Ypred == 1, 1], '.', color='lightblue')
    plt.plot(Xtest[Ypred == 2, 0], Xtest[Ypred == 2, 1], '.', color='mistyrose')
    plt.plot(Xapp[Yapp == 1, 0], Xapp[Yapp == 1, 1], '.b')
    plt.plot(Xapp[Yapp == 2, 0], Xapp[Yapp == 2, 1], '.r')

    # tracer la droite séparation et les marges...
    # droite : w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b)/w2
    # marges : w^T x + b = +1 et w^T x + b = -1
    w = model.coef_[0]
    b = model.intercept_[0]
    x1 = np.array([-5, 5])
    if abs(w[1]) > 1e-12:
        plt.plot(x1, -(w[0]*x1 + b) / w[1], 'k-')
        plt.plot(x1, -(w[0]*x1 + b - 1) / w[1], 'k--')
        plt.plot(x1, -(w[0]*x1 + b + 1) / w[1], 'k--')
    else:	# droite verticale x1 = -b/w1
        x2 = np.array([-5, 5])
        plt.plot([-b/w[0]]*2, x2, 'k-')
        plt.plot([(1-b)/w[0]]*2, x2, 'k--')
        plt.plot([(-1-b)/w[0]]*2, x2, 'k--')

    # entourer les vecteurs support
    sv = model.support_vectors_
    plt.plot(sv[:, 0], sv[:, 1], 'ko', fillstyle='none', markersize=8)
    plt.axis([-5, 5, -5, 5])

    # calculer et afficher la marge Delta...
    # distance entre l'hyperplan et le bord de la marge : Delta = 1 / ||w||
    Delta = 1 / np.linalg.norm(w)
    nbSV = len(sv)
    print("w = ", w, " b = ", b)
    print("Marge Delta = ", Delta, " (largeur totale 2*Delta = ", 2*Delta, ")")
    print("Nombre de vecteurs support = ", nbSV, model.n_support_)
    plt.title("C = " + str(C) + "   Delta = " + str(round(Delta, 3)) + "   #SV = " + str(nbSV))

    # pour réellement mettre à jour le graphique: (à garder en fin de fonction)
    fig.canvas.draw()


def monprogrammeNL(Xapp, Yapp, C, sigma):
    """
        Programme pour les SVM non linéaires (lancé avec N)
        
        Xapp, Yapp : base d'apprentissage générée avec la souris
        C	: paramètre réglé par +/-
        sigma : paramètre réglé par CTRL +/-
    """
    print("Apprentissage lancé avec " + str(len(Xapp)) + " points, C = ", C, " et sigma = ", sigma )

    # à compléter pour apprendre le modèle SVM non linéaire...
    model = svm.SVC(C=C, kernel='rbf', gamma=1/(2*sigma**2))
    model.fit(Xapp, Yapp)
    
    # création d'une grille de points de test
    r1 = np.arange(-5,5,0.2)
    Xtest = np.zeros((len(r1)*len(r1),2))
    i = 0
    for x1 in r1:
        for x2 in r1:
            Xtest[i,:] = [x1, x2]
            i += 1
    
    # Prédire la catégorie pour tous les points de test...
    Ypred = model.predict(Xtest)
    # ... et tracer le résultat avec par exemple 
    plt.cla()
    plt.plot(Xtest[Ypred == 1, 0], Xtest[Ypred == 1, 1], '.', color='lightblue')
    plt.plot(Xtest[Ypred == 2, 0], Xtest[Ypred == 2, 1], '.', color='mistyrose')
    plt.plot(Xapp[Yapp == 1, 0], Xapp[Yapp == 1, 1], '.b')
    plt.plot(Xapp[Yapp == 2, 0], Xapp[Yapp == 2, 1], '.r')


    # afficher le nombre de vecteurs support...	
    nsv = model.n_support_

    print("Vecteurs support par classe :", nsv, " total :", nsv.sum())
    plt.title("C = " + str(C) + ", sigma = " + str(sigma) + "  ->  " + str(nsv.sum()) + " vecteurs support")

    
    
    # pour réellement mettre à jour le graphique: 
    fig.canvas.draw()
    



##### Gestion de l'interface graphique ########


Xplot = np.zeros((0,2))
Yplot = np.zeros(0)
plotvariance = 0

C = 1
sigma = 1

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
    global C
    global sigma
    if event.key == " ":
        monprogramme(Xplot, Yplot, C)
    elif event.key == "n":
        monprogrammeNL(Xplot, Yplot, C, sigma)
    elif event.key == "i":
        # bascule entre marge dure (C = inf, cas séparable) et C = 10 (marge souple)
        C = 10 if C == math.inf else math.inf
        print("C = " , C)
    elif event.key == "+":
        C *= 2
        print("C = " , C)
    elif event.key == "-":
        C /= 2
        print("C = " , C)
    elif event.key == "ctrl++":
        sigma *= 2
        print("sigma = " , sigma)
    elif event.key == "ctrl+-":
        sigma /= 2
        print("sigma = " , sigma)
                
    
fig = plt.figure()

plt.axis([-5, 5, -5, 5])

cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("scroll_event", onscroll)
cid3 = fig.canvas.mpl_connect("key_press_event", onkeypress)

print("Utilisez la souris pour ajouter des points à la base d'apprentissage :")
print(" clic gauche : points bleus")
print(" clic droit : points rouges")
print("\nMolette : +/- variance ")
print("   si variance = 0  => ajout d'un point")
print("   si variance > 0  => ajout de points selon une loi gaussienne")
print("\n ESPACE pour lancer la fonction monprogramme(Xapp,Yapp,C)")
print("    avec la valeur de C modifiée par +/-")
print("    (touche I : bascule C = inf / C = 10)")
print("\n N pour lancer la fonction monprogrammeNL(Xapp,Yapp,C,sigma)")
print("    avec la valeur de C modifiée par +/-")
print("    et celle de sigma modifiée par CTRL +/-\n\n") 

plt.show()
