import cv2
import HandTrackingModule as htm
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

detector = htm.handDetector(maxHands=1)

path = '_27Americansignclassifier/data'
myList = os.listdir(path)

data = []
labels = []
for cls in myList:
    for img in os.listdir(f'{path}/{cls}'):
        img = cv2.imread(f'{path}/{cls}/{img}')
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            _x = [] 
            _y = []
            data_img = []
            for i in range(len(lmList)):
                _x.append(lmList[i][1])
                _y.append(lmList[i][2])
            
            for i in range(len(lmList)):
                # we append normalize data not row data
                data_img.append(lmList[i][1] - min(_x))
                data_img.append(lmList[i][2] - min(_y))
                
            data.append(data_img)
            labels.append(cls) # give only class name
         
# label encoding
label = LabelEncoder()
_labels = label.fit_transform(labels) # 0-35 for 0-9 and A-Z

data = np.asarray(data)
labels = np.asarray(_labels)

# train, test split
x_train, x_test, y_train, y_test = train_test_split(data, _labels, stratify=_labels, test_size=0.2, shuffle=True)

# model training
model = RandomForestClassifier()
model.fit(x_train, y_train)

# testing
y_pred = model.predict(x_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))

# model saving
pickle.dump(model, open('_27Americansignclassifier\model.pkl', 'wb'))
