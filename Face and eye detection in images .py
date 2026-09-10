import cv2
import matplotlib.pylab as plt 
#Import your photo
img=cv2.imread(r"C:\Users\Eye Net\Desktop\codes\IMG_20240709_201740_604.jpg")
eyes_model=cv2.CascadeClassifier(r"C:\Users\Eye Net\Desktop\aiolearn.py\ai tools\eyes.xml")
face_model=cv2.CascadeClassifier(r"C:\Users\Eye Net\Desktop\aiolearn.py\ai tools\haarcascade_frontalface_alt.xml")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
eyes=eyes_model.detectMultiScale(
    gray
)
face=face_model.detectMultiScale(
    gray
)

x=eyes[0][0]
y=eyes[0][1]
x2=x+ eyes[0][2]
x3=y+ eyes[0][3]

khat=cv2.rectangle(gray,(x,y),(x2,x3),(0,255,0),4) 

a=eyes[1][0]
b=eyes[1][1]
a2=a+ eyes[1][2]
a3=b+ eyes[1][3]
khat=cv2.rectangle(gray,(a,b),(a2,a3),(255,0,0),4)


z=face[0][0]
c=face[0][1]
z2=z+ face[0][2]
z3=c+ face[0][3]
khat=cv2.rectangle(gray,(z,c),(z2,z3),(0,0,255),2)



plt.imshow(khat)
plt.show() 













