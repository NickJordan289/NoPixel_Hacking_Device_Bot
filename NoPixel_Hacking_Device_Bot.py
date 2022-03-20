import numpy as np # Math
import cv2 # Computer vision 
import pyautogui # Button pressing
import mss # Screen grabbing
import ctypes # Windows Get Resolution

def main():
    while True:
        zone = {"left": 682, "top": 474, "width": 540, "height": 534}
        with mss.mss() as sct:
            base = np.array(sct.grab(zone))
            
        symbols_crop = base[163:187, 179:365]
        
        base = base[230:500, 70:465]
        base_draw = base.copy()
        
        detections = []
        for i in range (4):
            buf = 7
            symbol = symbols_crop[0:27, (40*i)+(buf*i):(40*(i+1))+(buf*i)]
            
            # resize symbol to 25x15
            symbol = cv2.resize(symbol, (25, 15))
            
            # perform template match of template onto base
            res = cv2.matchTemplate(base, symbol, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            detections.append(loc)
            
        # draw boxes around matches
        if len(detections) > 0:
            for detection in detections:
                for pt in zip(*detection[::-1]):
                    cv2.rectangle(base_draw, pt, (pt[0] + symbol.shape[1], pt[1] + symbol.shape[0]), (0,255,0), 2)
    
        cv2.imshow('base', base_draw)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()