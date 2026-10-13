import numpy as np
import matplotlib.pyplot as plt
import math

def kernel(X1,X2,sigma):
    """
        Retourne la matrice de noyau K telle que K_ij = K(X1[i], X2[j])
        avec un noyau gaussien K(x,x') = exp(-||x-x'||^2 / 2sigma^2)
    """	
    m1 = X1.shape[0]
    m2 = X2.shape[0]
    K = np.zeros((m1,m2))
    for i in range(m1):
        for j in range(m2):
            K[i,j] = math.exp(- np.linalg.norm(X1[i] - X2[j])**2 / (2*sigma**2))
    return K

def krrapp(X,Y,Lambda,sigma):
    """
        Retourne le vecteur beta du modèle Kernel ridge regression
        à noyau gaussien
        à partir d'une base d'apprentissage X,Y 
    """
    I = X.shape[0]
    K = kernel(X,X,sigma)
    beta =  np.linalg.inv(K + Lambda*np.eye(I))@Y 
    #beta = np.linalg.solve(K + Lambda*np.eye(m), Y)
    return beta
    
def krrpred(Xtest,Xapp,beta,sigma):
    """ 
        Retourne le vecteur des prédictions du modèle
        KRR à noyau gaussien de paramètres beta et sigma
    """
    Ktest = kernel(Xtest, Xapp, sigma)
    ypred = Ktest @ beta 
    
    return ypred
    

def kppvreg(Xtest, Xapp, Yapp, K):
    n = Xtest.shape[0]  # nb de points de test
    m = Xapp.shape[0]   # nb de points d'apprentissage
    ypred = np.zeros(n)

    # à compléter...
            
    return ypred





#################################################
#### Programme principal ########################
#################################################

## 1) générer une base de données de 1000 points X,Y

m = 1000
X = 6 * np.random.rand(m) - 3
Y = np.sinc(X) + 0.2 * np.random.randn(m)


# 2) Créer un base d'apprentissage (Xapp, Yapp) de 30 points parmi ceux de (X,Y) et une base de test(Xtest,Ytest) avec le reste des données

indexes = np.random.permutation(m)  # permutation aléatoire des 1000 indices entre 0 et 1000 
indexes_app = indexes[:30]  # 30 premiers indices
indexes_test = indexes[30:] # le reste

Xapp = X[indexes_app]
Yapp = Y[indexes_app]

Xtest = X[indexes_test]
Ytest = Y[indexes_test]

# ordronner les Xtest pour faciliter le tracé des courbes
idx = np.argsort(Xtest)
Xtest = Xtest[idx]
Ytest = Ytest[idx]

# tracer la figure

plt.figure()
plt.plot(Xtest,Ytest,'.r')
plt.plot(Xapp,Yapp,'*b')
plt.plot(Xtest,np.sinc(Xtest) , 'g')
plt.legend(['base test', 'base app', 'f_reg(x)'] )


### Tests de la Kernel ridge regression... 
   


for sigma in [0.1, 0.5, 2, 10]:
    for Lambda in [0.0001, 0.1, 10]:
        beta = krrapp(Xapp, Yapp, Lambda, sigma)
        ypred_app = krrpred(Xapp, Xapp, beta, sigma)
        ypred_test = krrpred(Xtest, Xapp, beta, sigma)

        plt.figure()
        
        plt.plot(Xtest, np.sinc(Xtest), 'g')
        plt.plot(Xtest, ypred_test, 'k', linewidth=2)
        plt.legend(['base test', 'base app', 'f_reg(x)', 'f(x) KRR'])
        plt.title(f"sigma = {sigma}, lambda = {Lambda}")


# lorsqu'on augmente lamda le model soit moin complexe et donc moins de variance mais plus de biais.
#lorsque je diminue sigma on obtient beaucoup d'oscilation 
# petite sigma et petite lamda plus d'oscillation et donc plus de complexite cad sur-apprentissage 


# Influence de lambda :
#  - lambda grand : régularisation forte, modèle moins complexe (plus lisse, écrasé vers 0)
#   -sous-apprentissage,
#    -erreurs d'apprentissage ET de test élevées.
#  - lambda petit : le modèle colle aux points d'apprentissage (et donc au bruit).
#
# Influence de sigma :
#  - sigma petit : chaque noyau n'agit qu'autour de son point -> beaucoup d'oscillations,
#    et f(x) retombe vers 0 entre les points d'apprentissage.
#  - sigma grand : modèle très lisse, qui ne suit plus les oscillations du sinc -> sous-apprentissage.
#
# sigma petit et lambda petit : modèle très complexe qui passe par tous les points
#   -> erreur d'apprentissage très faible mais erreur de test élevée : sur-apprentissage.
#
# Le meilleur compromis (ici vers sigma = 0.5, lambda = 0.1 ou 0.01) donne l'erreur de test
# la plus faible, proche de la variance du bruit (0.2^2 = 0.04), qu'on ne peut pas descendre en dessous.
# Une erreur d'apprentissage faible ne garantit donc pas une bonne généralisation.

### Tests avec les Kppv...





# Affichage des graphiques : 
# (à ne faire qu'en fin de programme)
plt.show() # affiche les plots et bloque en attendant la fermeture de la fenêtre

