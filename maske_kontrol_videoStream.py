from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# çerçevenin boyutlarını ayarlama kısmı
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# yüz algılama kısmı
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# face listesi, face konumu, ve face maskesi ağımızdaki tahminlerin listesi tanımlıyoruz
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		# tespit ile ilişkili güveni (olasılık) çıkarmak
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# en az bir yüz algılandıysa bir tahmin yap
	if len(faces) > 0:
		# Daha hızlı çıkarım için, yukarıdaki "for" döngüsündeki tek tek tahminler yerine
		# * tüm * yüzler için aynı anda toplu tahminler yapacağız
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

# yüz dedektörü modelimizi 'face_detector' klasörden yüklüyoruz
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# oluşturduğumuz 'mask_detector.model' dosyayı yani yüz maske dedektörü yüklüyoruz
maskNet = load_model("mask_detector.model")

# video stream başlat
print("[INFO] Video stream başlıyor...")
vs = VideoStream(src=0).start()

while True:
	# video stream'den gelen görüntünün genişliğini maksimum 400 pixel olarak ayarlıyoruz.
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# insanların yüzleri algılama ve yüz maskeli olup olmadığını belirleme kısmı
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# İnsan yüzünü ayırt eden çerçeve ve metni çizmek için kullanacağımız sınıf etiketini 
		# ve rengini belirliyoruz
		label = "Maskeli" if mask > withoutMask else "Maskesiz"
		color = (0, 255, 0) if label == "Maskeli" else (0, 0, 255)

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# "q" tuşuna basıldıysa döngüden çık
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()