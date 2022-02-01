from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# ilk öğrenme hızı (1e-4 = 0.0001)
ogrenme_hiz = 1e-4 
# Epoch(döngü) sayısı, eğitim sırasında tüm eğitim verilerinin ağa gösterilme sayısıdır.
EPOCHS = 20 
 # batch size(boyutu), parametre güncellemesinin gerçekleştiği ağa verilen alt örneklerin sayısıdır.
Batch_boyutu = 32

resimlerYolu = r"C:\Users\windows10\Desktop\Covid_19 maske kontrol\resimler"
maskeKategoriler = ["maskeli", "maskesiz"]

print("[INFO] resimler yukleniyor...")

resimArray = []
kategoriArray = []

for kategori in maskeKategoriler: #öncelikle maskeli resimlere, sonra maskesiz resimlere bakar
    path = os.path.join(resimlerYolu, kategori)
    for res in os.listdir(path):
    	resim_path = os.path.join(path, res) # maskeli ve maskesiz resimleri bir listeye birleştirip atıyor
    	resim = load_img(resim_path, target_size=(224, 224)) # resimleri 224x224 pixel boyutuna ayarlayıp yüklüyör
    	resim = img_to_array(resim) # resimleri diziye atıyor
    	resim = preprocess_input(resim) # görüntü ön işleme yapiyor

    	resimArray.append(resim)
    	kategoriArray.append(kategori)

lb = LabelBinarizer()
kategoriArray = lb.fit_transform(kategoriArray)
kategoriArray = to_categorical(kategoriArray)

resimArray = np.array(resimArray, dtype="float32")
kategoriArray = np.array(kategoriArray)

(trainX, testX, trainY, testY) = train_test_split(resimArray, kategoriArray,
	test_size=0.20, stratify=kategoriArray, random_state=42)

resimGenerator = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

#oluşturduğumuz modelimizin derleme (compile) işlemi
print("[INFO] compiling model...")
opt = Adam(lr=ogrenme_hiz, decay=ogrenme_hiz / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# ağın head kısmını eğitme işlemi
print("[INFO] head kısmını eğitme işlemi gerçekleşiyor...")
H = model.fit(
	resimGenerator.flow(trainX, trainY, batch_size=Batch_boyutu),
	steps_per_epoch=len(trainX) // Batch_boyutu,
	validation_data=(testX, testY),
	validation_steps=len(testX) // Batch_boyutu,
	epochs=EPOCHS)

# test setiyle ilgili tahminler
print("[INFO] ağın değerlendirme işlemi gerçekleşiyor...")
predIdxs = model.predict(testX, batch_size=Batch_boyutu)

# Test setindeki her bir görüntü için 
# karşılık gelen en büyük tahmini olasılıkla etiketin dizinini bulmamız gerekir.
predIdxs = np.argmax(predIdxs, axis=1)

# sınıflandırma raporu göster
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# eğitilmiş modelimizi klasöre dosya olarak kaydediyoruz
print("[INFO] maske detector modelini kaydediyoruz...")
model.save("mask_detector.model", save_format="h5")

# eğitim kaybını(loss) ve doğruluğunu(accuracy) çizme kodları
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")