import cv2
import numpy as np
import face_recognition
#import time
from datetime import datetime
from pyzbar.pyzbar import decode
import qrcode
import os
# from PIL import ImageGrab

 #已经知道脸的代码编程
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 #
def markAttendance(name):
    with open('facedata/Attendance.csv','w+') as f:#‘\’ 是转义符号
        myDataList = f.readlines()#[]
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:#记录开始出现的时间
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
       
   # print(data_string)
def make_QR(medicine_person):
    qr = qrcode.QRCode(
        version=None ,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
        )
    qr.add_data('XUCHEN')
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save("facedata/QRcode.bmp")



medicine_person = 'XUCHEN'
make_QR(medicine_person)

path = 'facedata/image'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
     
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
  #已经知道脸的代码编程
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)

success = 1
while success:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#多脸识别
    
    facesCurFrame = face_recognition.face_locations(imgS)
    #多脸编码
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)#最匹配
 
        if matches[matchIndex]:#和match一样的时候
            name = classNames[matchIndex].upper()#方法将字符串中的小写字母转为大写字母
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            success = 0
            cv2.imshow('Webcam',img)
            cv2.waitKey(200)
            
cap.release()
cv2.destroyAllWindows() 
    
#make_QR

cap2 = cv2.VideoCapture(0)

while success == 0:
    success, img = cap2.read()
    #img = captureScreen()
    success = 0
    imgS = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('Webcam2',imgS)
    cv2.waitKey(1)
#recognation QR
    data = decode(imgS)
    if len(data) != 0:
        data_string = data[0][0].decode("utf-8")
        if data_string == name:
            print("please take the medicine.")
        else:
            print("please check the medicine.")
    data = []
    # os.system('pause')



        
    
    