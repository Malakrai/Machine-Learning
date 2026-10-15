import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score


#### programme principal à implémenter dans cette fonction ####
def monprogramme(X, K):
    """
        X : base d'apprentissage générée avec la souris
        K : paramètre réglé par +/-
    """
    print("Kmeans clustering lancé avec " + str(len(X)) + " points et K = ", K)

    clustering = KMeans(n_clusters=K, init='random', n_init=100)
    clustering.fit(X)

    y = clustering.labels_
    centres = clustering.cluster_centers_
    print("Coût (inertie) :", clustering.inertia_)

    # tracé du résultat
    plt.figure(1)
    plt.clf()
    plt.axis([-5, 5, -5, 5])

    couleurs = ['b','r','g','y','m','c']

    # points de chaque groupe
    for k in range(K):
        Xk = X[y == k]
        plt.plot(Xk[:,0], Xk[:,1], '.' + couleurs[k % len(couleurs)])

    # centres avec un autre type de point
    for k in range(K):
        plt.plot(centres[k,0], centres[k,1], 'x' + couleurs[k % len(couleurs)],
                 markersize=15, markeredgewidth=3)

    # pour réellement mettre à jour le graphique:
    fig.canvas.draw()




### programme pour Spectral clustering ###
def monprogrammeSpectralClustering(X, K, sigma):
    print("Spectral Clustering lancé avec " + str(len(X)) + " points et sigma = ", sigma)

    clustering = SpectralClustering(n_clusters=K, affinity='rbf', gamma=1/(2*sigma**2))
    clustering.fit(X)
    y = clustering.labels_
    A = clustering.affinity_matrix_

    # figure 1 : points colorés
    plt.figure(1)
    plt.clf()
    plt.axis([-5, 5, -5, 5])
    couleurs = ['b','r','g','y','m','c','k']
    for k in range(K):
        plt.plot(X[y == k, 0], X[y == k, 1], '.' + couleurs[k % len(couleurs)])

    # figure 2 : matrice d'affinités
    fig2 = plt.figure(2)
    plt.clf()
    ordre = np.argsort(y)                    # trie les points par groupe
    plt.imshow(A[np.ix_(ordre, ordre)], cmap='gray')
    plt.title("Matrice d'affinités A (scikit-learn)")
    fig2.canvas.draw()
    fig2.show()

    # comparaison avec notre propre implémentation (Lsym)
    y2 = monSpectralClustering(X, K, sigma)
    print("Accord avec scikit-learn (ARI) :", adjusted_rand_score(y, y2))

    plt.figure(1)        # pour que les clics s'ajoutent bien à la figure principale
    fig.canvas.draw()


### version personnelle du spectral clustering (laplacien normalisé Lsym) ###
def kernel(X1, X2, sigma):
    # noyau gaussien : K_ij = exp(-||x_i - x_j||^2 / 2 sigma^2)
    dist2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    return np.exp(-np.maximum(dist2, 0) / (2 * sigma**2))


def monSpectralClustering(X, K, sigma):
    # matrice d'affinités avec diagonale nulle
    A = kernel(X, X, sigma)
    A = A - np.diag(np.diag(A))

    # matrice des degrés et D^{-1/2}
    D = np.diag(np.sum(A, axis=1))
    Dm12 = np.sqrt(np.linalg.inv(D))

    # laplacien normalisé Lsym = I - D^{-1/2} A D^{-1/2}
    Lsym = np.eye(len(X)) - Dm12 @ A @ Dm12

    # K vecteurs propres associés aux K plus petites valeurs propres
    valp, vecp = np.linalg.eigh(Lsym)        # valeurs propres triées par ordre croissant
    U = vecp[:, :K]

    # normalisation des lignes (Ng, Jordan, Weiss)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)

    # K-means dans l'espace spectral
    y = KMeans(n_clusters=K, n_init=10).fit(U).labels_

    # figure 3 : résultat de notre implémentation
    fig3 = plt.figure(3)
    plt.clf()
    plt.axis([-5, 5, -5, 5])
    couleurs = ['b','r','g','y','m','c','k']
    for k in range(K):
        plt.plot(X[y == k, 0], X[y == k, 1], '.' + couleurs[k % len(couleurs)])
    plt.title("Spectral clustering maison (Lsym), sigma = " + str(sigma))
    fig3.canvas.draw()
    fig3.show()

    return y

##### Gestion de l'interface graphique ########


Xplot = np.zeros((0,2))
plotvariance = 0

K = 2
sigma = 1

def onclick(event):
    global Xplot
    
    if plotvariance == 0:
        newX = np.array([[event.xdata,event.ydata]])
    else:
        newX = math.sqrt(plotvariance) * np.random.randn(10, 2) + np.ones((10,1)).dot(np.array([[event.xdata,event.ydata]]))

    print("Ajout de " + str(len(newX)) + " points en (" + str(event.xdata) + ", " + str(event.ydata) + ")")

    Xplot = np.concatenate((Xplot,newX))
    if event.button == 1 and event.key == None:
        plt.plot(newX[:,0], newX[:,1],'.k')
    
    fig.canvas.draw()


def onscroll(event):
    global plotvariance
    if event.button == "up":
        plotvariance = round(plotvariance + 0.2, 1)
    elif event.button == "down" and plotvariance > 0.1:
        plotvariance = round(plotvariance - 0.2, 1)
    print("Variance = ", plotvariance)

def onkeypress(event):
    global K
    global sigma
    global Xplot
    if event.key == " ":
        monprogramme(Xplot, K)
    elif event.key == "c":
        monprogrammeSpectralClustering(Xplot,K,sigma)
    elif event.key == "+" and K < len(Xplot):
        K += 1
        print("K = " , K)
    elif event.key == "-" and K > 1:
        K -= 1
        print("K = " , K)
    elif event.key == "ctrl++":
        sigma *= 2
        print("sigma = " , sigma)
    elif event.key == "ctrl+-":
        sigma /= 2
        print("sigma = " , sigma)
    elif event.key == 'delete':
        Xplot = np.zeros((0,2))	
        plt.clf()
        plt.axis([-5, 5, -5, 5])
        fig.canvas.draw()
    
fig = plt.figure()

plt.axis([-5, 5, -5, 5])

cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("scroll_event", onscroll)
cid3 = fig.canvas.mpl_connect("key_press_event", onkeypress)

print("Utilisez la souris pour ajouter des points à la base d'apprentissage :")
print(" clic gauche : points ")
print("\nMolette : +/- variance ")
print("   si variance = 0  => ajout d'un point")
print("   si variance > 0  => ajout de points selon une loi gaussienne")
print("\n ESPACE pour lancer la fonction monprogramme(X,K)")
print("    avec la valeur de K modifiée par +/-\n\n") 
print("\n C pour lancer la fonction monprogrammeSpectralClustering(X,K,sigma)")
print("    avec la valeur de sigma modifiée par Ctrl +/-\n\n") 

plt.show()
