from math import dist
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
            
        positions = []
        # draw boxes around matches
        if len(detections) > 0:
            for detection in detections:
                # draw circle at center of detected symbol
                for pt in zip(*detection[::-1]):
                    x = int(int(pt[0]) + (symbol.shape[0]/2))
                    y = int(int(pt[1]) + (symbol.shape[1]/2))
                    positions.append((x, y))
        
        for position in positions:
            positions_copy = positions.copy()
            positions_copy.remove(position)
            
            min_d = 10000
            for p in positions_copy:
                d = dist(position, p)
                if d < min_d:
                    min_d = d
            
            if min_d <= 45:
                cv2.circle(base_draw, position, 5, (0,255,0), -1)
            
        cv2.imshow('base', base_draw)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()