import cv2
import numpy as np
import math
def callme():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
                ret , frame = cap.read()
                cv2.rectangle(frame,(100,100), (300,300),(0,255,0),0)
                ci = frame[100:300,100:300]
                blur=cv2.GaussianBlur(ci,(3,3),0)
                hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
                mask2 =cv2.inRange(hsv, np.array([2,0,0]),np.array([20,255,255]))
                kl=np.ones((5,5))
                dt=cv2.dilate(mask2,kl,iterations=1)
                eros=cv2.erode(dt,kl,iterations=1)
                fl=cv2.GaussianBlur(eros,(3,3),0)
                ret,thresh=cv2.threshold(fl,127,255,0)
                cv2.imshow("threshold image",thresh)
                image,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                try:
                    contour=max(contours,key=lambda x: cv2.contourArea(x))
                    x,y,w,h=cv2.boundingRect(contour)
                    cv2.rectangle(ci,(x,y),(x+w,y+h),(0,0,255),0)
                    hull=cv2.convexHull(contour)
                    hp = np.zeros(ci.shape,np.uint8)
                    cv2.drawContours(hp,[contour],-1,(0,255,0),0)
                    cv2.drawContours(hp,[hull],-1(0,0,255),0)
                    hull=cv2.convexHull(contour,returnPoints=False)
                    defects=cv2.convexityDefects(contour,hull)
                    count_defects=0
                    for i in range(defects.shape[0]):
                        s,e,f,d=defects[i,0]
                        start=tuple(contour[s][0])
                        end=tuple(contour[e][0])
                        far=tuple(contour[f][0])
                        a=math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
                        b=math.sqrt((far[0]-start[0])**2+(far[1]-start[1])**2)
                        c=math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
                        angle=(math.acos((b**2+c**2-a**2)/(2*b*c))*180)/3.14
                        if angle<=90:
                            count_defects+=1
                            cv2.circle(ci,far,1,[0,0,155],-1)
                        cv2.line(ci,start,end,[0,255,0],2)
                    if count_defects==4:
                        cv2.putText(frame,'Five number or fingure',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                    elif count_defects==3:
                        cv2.putText(frame,'Four number or fingure',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                    elif count_defects==2:
                        cv2.putText(frame,'Three number or fingure',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                    elif count_defects==1:
                        cv2.putText(frame,'Two number or fingure',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                    elif count_defects==0:
                        cv2.putText(frame,'one number or fingure',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                    else:
                        pass
                except:
                    pass
                    
                cv2.imshow("hand Gesture",frame)
                all_image=np.hstack((hp,ci))
                cv2.imshow('counter is: ',all_image)
                if cv2.waitKey(1) == ord('s'):
                    break
        cv2.realse()
        cv2.destroyAllWindows()


callme()
