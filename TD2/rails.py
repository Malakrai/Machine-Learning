from sklearn import svm
import numpy as np
import time

data = np.loadtxt("defautsrails.dat")
X = data[:,:-1] # tout sauf la dernière colonne
y = data[:,-1] # uniquement la dernière colonne

models = []
C = 1

#boucle pour créer un classifieur SVM pour chaque classe (1 à 4)
for k in range(1, 5):
    yk = 2*(y==k) - 1
    model = svm.SVC(C=C, kernel='linear')
    model.fit(X, yk)
    models.append(model)

    # taux de reconnaissance sur la base d'apprentissage
    taux = np.mean(model.predict(X) == yk) * 100
    print("Classifieur", k, ":", taux, "%")

G = np.zeros((4, len(X)))
for k in range(1, 5):
    G[k-1,:] = models[k-1].decision_function(X)

y_pred =np.argmax(G, axis=0)+1
#erreur
erreur = np.mean(y_pred != y) * 100
print("Erreur d'apprentissage multi-classe :", erreur, "%")




debut = time.time()
#la validation croisée
def ErrLOO(X, y, C):  
    erreur_validation = 0


    for i in range(len(y)):
        X_i = np.delete(X, i, axis=0)
        y_i = np.delete(y, i)

        models_i = []
        for k in range(1, 5):
            yk = 2*(y_i==k) - 1
            model = svm.SVC(C=C, kernel='linear')
            model.fit(X_i, yk)
            models_i.append(model)
        G = np.zeros(4)
        for k in range(1, 5):
            G[k-1] = models_i[k-1].decision_function([X[i]])[0]

        y_pred = np.argmax(G) + 1
        if y_pred != y[i]:
            erreur_validation += 1

    return erreur_validation / len(y) * 100

    #erreur_validation = erreur_validation / len(y) * 100
    #print("Erreur de validation croisée :", erreur_validation, "%")


erreur = ErrLOO(X, y, 1)
fin = time.time()
print("Erreur LOO :", erreur, "%")
print("Temps de calcul :", fin - debut, "secondes")

C=[0.01,0.1,1,5,10,100]
erreurs = []

for c in C:
   erreur= ErrLOO(X, y, c)
   erreurs.append(erreur)
   print("C =", c, " -> erreur LOO =", erreur, "%")

meilleur_C = C[np.argmin(erreurs)]
print("Meilleur C :", meilleur_C, " avec une erreur de", min(erreurs), "%")

meuilleur_model = svm.SVC(C=meilleur_C, kernel='linear')


# 3.4.1 : validation croisée imbriquée (LOO dans LOO) 
C = [0.01, 0.1, 1, 5, 10, 100]
debut = time.time()
nb_erreurs = 0

for i in range(len(y)):                      # Pour chaque (xi, yi)
    
    X_i = np.delete(X, i, axis=0)            # les 139 exemples
    y_i = np.delete(y, i)

    erreurs_C = []
    for c in C:                             
        erreurs_C.append(ErrLOO(X_i, y_i, c))    # LOO2 sur les 139 exemples

    meilleur_C = C[np.argmin(erreurs_C)]     

    # meilleur modèle = appris sur les 139 exemples avec meilleur C
    G = np.zeros(4)
    for k in range(1, 5):
        yk = 2*(y_i==k) - 1
        model = svm.SVC(C=meilleur_C, kernel='linear')
        model.fit(X_i, yk)
        G[k-1] = model.decision_function([X[i]])[0]   # test sur (xi, yi)

    if np.argmax(G) + 1 != y[i]:
        nb_erreurs += 1

print("ErrLOO finale :", nb_erreurs / len(y) * 100, "%")   
print("Temps :", time.time() - debut, "secondes")