import cv2
import numpy as np
import utlis
scale = 3
wP=210*scale
hP=297*scale


######

webcam = False
path = '1.jpeg'
# path = '5.jpeg'
cap = cv2.VideoCapture(0)
cap.set(10,0 ) #Id for Brightness

cap.set(3,1920) #Id for Width

cap.set(4,1080) #Id for Height

while True:
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)

    img, conts = utlis.getContours(img,showCanny=False,minArea=50000,filter=4,draw=True)

    if len(conts)!=0:
        biggest = conts[0][2]
        # print(biggest)
        imgWarp = utlis.warpImg(img,biggest,wP,hP)

        img2, conts2 = utlis.getContours(imgWarp, showCanny=True, minArea=2000, filter=4, draw=False, cThr= [50,50])
        if len(conts2)!=0:
            for obj in conts2:
                cv2.polylines(img2,[obj[2]],True,(0,255,0),2)
                nPoints = utlis.reorder(obj[2])
                nW=round((utlis.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                nH = round((utlis.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)


                cv2.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(img2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(img2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)


        cv2.imshow('Warped Image', img2)

    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    cv2.waitKey(1)