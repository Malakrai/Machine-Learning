import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

images = ['china.jpg', 'flower.jpg']

# Question 3 : différentes valeurs de K testées
valeursK = [2, 4, 8, 16]

# tests sur les 2 images
for image in images:
	img = load_sample_image(image)

	(width,height,channels) = img.shape

	# Question 1 : création de la matrice des pixels
	# une ligne par pixel, une colonne par canal (R, V, B)
	X = img.reshape(width * height, channels)

	# figure : originale + étiquettes + segmentation pour chaque K
	plt.figure(figsize=(4 * (len(valeursK) + 1), 7))
	plt.subplot(2, len(valeursK) + 1, 1)
	plt.imshow(img)
	plt.title('Originale')
	plt.axis('off')

	for i, K in enumerate(valeursK):
		# Question 2 : K-means sur les pixels
		kmeans = KMeans(n_clusters=K, n_init=4, random_state=0)
		etiquettes = kmeans.fit_predict(X)

		# récupération des étiquettes sous forme matricielle :
		Y = etiquettes.reshape(width, height)

		# Création de l'image couleur à 3 canaux :
		# chaque pixel prend la couleur du centre de son groupe
		segmentation = kmeans.cluster_centers_[etiquettes].reshape(width, height, channels)

		# afficher les étiquettes Y et la segmentation avec la couleur moyenne :
		plt.subplot(2, len(valeursK) + 1, i + 2)
		plt.imshow(Y)
		plt.title('Etiquettes, K = %d' % K)
		plt.axis('off')
		plt.subplot(2, len(valeursK) + 1, len(valeursK) + i + 3)
		plt.imshow(segmentation / 255)
		# les centres sont à valeurs réelles,
		# donc imshow s'attend à avoir des valeurs entre 0 et 1 pour segmentation
		# au lieu de 0 et 255 pour les entiers
		plt.title('Segmentation, K = %d' % K)
		plt.axis('off')

	plt.suptitle(image)
	plt.tight_layout()
	plt.draw()

plt.show()
		
