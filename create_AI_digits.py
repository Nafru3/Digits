import tensorflow as tf
import keras as kr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def afficheImage(image_a_afficher, target):
    plt.imshow(np.reshape(image_a_afficher, (28, 28)), cmap="binary") # On précise l'image à afficher, la taille et binary sert à dire que c'est en niveau de gris
    plt.title(str(target))
    plt.show()

mnist = kr.datasets.mnist
(images, targets), (_, _) = mnist.load_data()
# print(x_train.shape) # On voit qu'il y a 60000 images de taille 28*28

images = images[:20000]
targets = targets [:20000]



images = images.reshape(-1, 784)
images = images.astype(float)


scaler = StandardScaler()

images = scaler.fit_transform(images) # Permet de normaliser
#image_test = scaler.fit_transform(image_test)

#afficheImage(images[0],targets[0])
#afficheImage(image_test[0],target_test[0])


model = tf.keras.models.Sequential() # On crée le modèle
model.add(tf.keras.layers.Dense(256, activation="relu")) 
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile( # On compile le modèle
    loss="sparse_categorical_crossentropy", 
    optimizer="sgd", # Descente de gradient
    metrics=["accuracy"]
)


history = model.fit(images, targets, epochs=20, validation_split=0.2) # batch_size est à 32 de base : Ca correspond par combien on divise pour le nombre de test (quand on met 8000 images ca donne 8000/32 = 500) plus le nombre est petit plus ça sera lent mais précis et inversement 

# On affiche les courbes
loss_curve = history.history["loss"] # Affiche courbe des erreurs
acc_curve = history.history["accuracy"] # Affiche courbe des pourcentages de bonnes valeurs

loss_val_curve = history.history["val_loss"] # On récupère la validation 
acc_val_curve = history.history["val_accuracy"] # Si validation commence à stagner à une valeur haute et ne redescend pas : C'est parfait !

model.save("ia_chiffre.keras")