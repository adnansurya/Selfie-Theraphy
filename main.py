import cv2
import os
import time
import numpy as np

cascPath = "haarcascade_frontalface_default.xml"
pengklasifikasiWajah = cv2.CascadeClassifier( os.path.join(cv2.data.haarcascades, cascPath) )

objVideo = cv2.VideoCapture(0)
if not objVideo.isOpened():
    print('Kamera tidak dapat diakses')
    exit()

tombolQDitekan = False
#objVideo.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#objVideo.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while (tombolQDitekan == False):
    ret, kerangka = objVideo.read()

    if ret == True:
        abuAbu = cv2.cvtColor(kerangka, cv2.COLOR_BGR2GRAY)
        dafWajah = pengklasifikasiWajah.detectMultiScale(abuAbu, scaleFactor=1.3, minNeighbors=2)
        if str(dafWajah) == '()':
            print('tidak ada wajah')
        else:
            #time.sleep(5)
            #ret, frame = objVideo.read()
            #image_name = 'muka.jpg'
            #cv2.imwrite(image_name, frame)
            #print(image_name)
            print('ada wajah')
            #break
        for(x,y,w,h) in dafWajah:
            cv2.rectangle(kerangka, (x,y), (x+y, y+h), (255,0,0), 2)

        cv2.imshow('Hasil', kerangka)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            tombolQDitekan =  True
            break
    else:
        break

objVideo.release()
cv2.destroyAllWindows()

