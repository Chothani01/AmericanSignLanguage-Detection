import cv2
import HandTrackingModule as htm
import numpy as np
import os
import time
import pickle

model = pickle.load(open('_27Americansignclassifier/model.pkl', 'rb'))
labels = []
for cls in os.listdir('_27Americansignclassifier/data'):
    labels.append(cls)
    
detector = htm.handDetector(maxHands=1, detectionConfidence=0.7)

cap = cv2.VideoCapture(0)
ptime = 0

while True:
    res, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        data=[]
        
        x = [point[1] for point in lmList]
        y = [point[2] for point in lmList]
        
        # Below code add normalize data in dataset
        for i in range(len(lmList)):
            data.append(lmList[i][1] - min(x))
            data.append(lmList[i][2] - min(y))
        
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(img, pt1=(xmin-20, ymin-20), pt2=(xmax+20, ymax+20), color=(255, 0, 0), thickness=3)
        
        # prediction
        data = np.asarray(data) 
        prediction = model.predict([data])
        print(prediction)
        label = labels[prediction[0]]
        cv2.putText(img, text=str(label), org=(xmin-20, ymin-40), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(255, 255, 0), thickness=2)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    
    cv2.putText(img, text=f"Text: {int(fps)}", org=(40, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 255), thickness=2)
    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF==ord('x'):
        break 
    
cap.release()
cv2.destroyAllWindows()
    

