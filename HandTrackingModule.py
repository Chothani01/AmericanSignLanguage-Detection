import cv2
import time
import mediapipe as mp
import math

# id with name for hand
## id ## Name
## 0  -  Base of the hand
## 1  -  Thumb carpometacarpal joint
## 2  -  Thumb metacarpophalangeal joint
## 3  -  Thumb interphalangeal joint
## 4  -  Thumb tip
## 5  -  Index metacarpophalangeal joint
## 6  -  Index proximal interphalangeal joint
## 7  -  Index distal interphalangeal joint
## 8  -  Index fingertip
## 9  -  Middle metacarpophalangeal joint
## 10 -  Middle proximal interphalangeal joint
## 11 -  Middle distal interphalangeal joint
## 12 -  Middle fingertip
## 13 -  Ring metacarpophalangeal joint
## 14 -  Ring proximal interphalangeal joint
## 15 -  Ring distal interphalangeal joint
## 16 -  Ring fingertip
## 17 -  Pinky metacarpophalangeal joint
## 18 -  Pinky proximal interphalangeal joint
## 19 -  Pinky distal interphalangeal joint
## 20 -  Pinky fingertip

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectionConfidence, 
                                        min_tracking_confidence=self.trackConfidence) # by default parameters
        self.mpDraw = mp.solutions.drawing_utils
        
        self.tipIds = [4, 8, 12, 16, 20] # [thumb tip, Index fingertip, Middle fingertip, Ring fingertip, pinky fingertip]
        
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                # if we not write mpHands.HAND_CONNECTIONS than we not see connection between landmarks
        return img
        
    
    def findPosition(self, img, handNo=0, draw=True):
        self.lmList=[]
        xList = []
        yList = []
        bbox= []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, center=(cx, cy), radius=12, color=(255, 0, 255), thickness=-1)
        
            xmin = min(xList)
            xmax = max(xList)
            ymin = min(yList)
            ymax = max(yList)
            bbox = xmin, ymin, xmax, ymax
        
            if draw:
                cv2.rectangle(img, pt1=(xmin-20, ymin-20), pt2=(xmax+20, ymax+20), color=(0, 255, 0), thickness=2)
        
        return self.lmList, bbox
    
    
    def fingerUps(self):
        fingers=[]
        
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
            
        else:
            fingers.append(0)
            
        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                # because (0, 0) is top left and (x_max, y_max) is bottom right
                fingers.append(1)
            
            else:
                fingers.append(0)
        
        return fingers  
    
    def findDistance(self, p1, p2, img, draw=True, r=12, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        
        cx, cy = (x1+x2)//2, (y1+y2)//2 
        
        if draw:
            cv2.line(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=t)
            cv2.circle(img, center=(x1, y1), radius=r, color=((255, 0, 255)), thickness=t)     
            cv2.circle(img, center=(x2, y2), radius=r, color=((255, 0, 255)), thickness=t)     
            cv2.circle(img, center=(cx, cy), radius=r, color=((0, 0, 255)), thickness=t) 
        length = math.hypot(x1-x2, y1-y2) # math.hypot(a, b) = (a^2 + b^2)^0.5
            
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    ptime = 0
    ctime = 0
    
    detector = handDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        res, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0:
            print(lmList[0])
            
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        
        cv2.putText(img, text=str(int(fps)), org=(10, 70), color=(255, 0 , 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF==ord('x'):
            break
        
if __name__=="__main__":
    main()