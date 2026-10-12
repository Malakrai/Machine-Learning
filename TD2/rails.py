from sklearn import svm
import numpy as np

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


C=[0.01,0.1,1,5,10,100]
erreurs = []

for c in C:
   erreur= ErrLOO(X, y, c)
   erreurs.append(erreur)
   print("C =", c, " -> erreur LOO =", erreur, "%")

meilleur_C = C[np.argmin(erreurs)]
print("Meilleur C :", meilleur_C, " avec une erreur de", min(erreurs), "%")

meuilleur_model = svm.SVC(C=meilleur_C, kernel='linear')

