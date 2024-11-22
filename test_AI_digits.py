import keras as kr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Charger image : 
image = Image.open("Pictures/test_image_ia_chiffre_3.png")
image = image.convert("L") # Convertit en niveau de gris
image = image.resize((28, 28)) # Redimensionne image
image = np.array(image) # On convertit en tableau mais le blanc correspon à 255 alors que dans la base de donnée c'était des 0
image = 255 - image # Pour que le blanc soit à 0

def afficheImage(image_a_afficher, target, prediction):
    plt.imshow(np.reshape(image_a_afficher, (28, 28)), cmap="binary") # On précise l'image à afficher, la taille et binary sert à dire que c'est en niveau de gris
    plt.title("La prédiction est : " + str(prediction) + "    Le vrai résultat est : " + str(target))
    plt.show()

mnist = kr.datasets.mnist
(_, _), (images_test, targets_test) = mnist.load_data()
# print(x_train.shape) # On voit qu'il y a 60000 images de taille 28*28

i = 14
#image_test = images_test[i]
target_test = targets_test[i]

image_test = image
target_test = 3

#print("image test : ", image_test)

image_test = np.expand_dims(image_test, axis=0) # Pour convertir en numpy.ndarray
target_test = np.expand_dims(target_test, axis=0)

image_test = image_test.reshape(-1, 784)
image_test = image_test.astype(float)

image_test = image_test / 255 # Car scaler.fit_transform(image_test) ne marche pas car StandarsScaler() fonctionne mieux s'il y a beaucoup de donnée (Car il calcule moyenne et ecart type)

model_loaded = kr.models.load_model("ia_chiffre.keras")
predictions = model_loaded.predict(image_test)
pred = np.argmax(predictions, axis=1)[0]
print("La prédiction est :", pred) 
afficheImage(image_test[0],target_test[0], pred)