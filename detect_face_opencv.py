import numpy as np
import cv2
import dlib
import winsound

face_cascade = cv2.CascadeClassifier('C:\\Users\\User\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\User\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')

test=face_cascade.load('C:\\Users\\User\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
print(test)
img = cv2.imread('face2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(10, 10))
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


predictor_path="C:\\Users\\User\\Anaconda3\\Lib\\site-packages\\dlib\\shape_predictor_68_face_landmarks.dat"
predictor=dlib.shape_predictor("C:\\Users\\User\\Anaconda3\\Lib\\site-packages\\dlib\\shape_predictor_68_face_landmarks.dat")

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
def turn_on_alarm():
    winsound.PlaySound('sound.wav', winsound.SND_FILENAME)

turn_on_alarm()

import numpy as np

a=np.array([[[9, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
#print (np.apply_along_axis(lambda h: h.searchsorted(45), axis = 1, arr=a))

#b=np.array([1, 3, 4])
#print(np.searchsorted(b, 4))
b = np.reshape(a, (np.product(a.shape),))
c=a.ravel()
e=np.sort(c)
print(e)
d=np.searchsorted(e, 0)
print(c)
print(d, len(e))

dmedian_blur=median_blur.ravel()
        print(np.sort(dmedian_blur))
        x=np.searchsorted(dmedian_blur, 0, sorter=None)
        if x==len(dmedian_blur):
            t+=1
        else:
            found=True
    return t
def slope(x_val, y_val):
    x=np.array(x_val)
    y=np.array(y_val)
    try:
        m=(((np.mean(x)*np.mean(y))-np.mean(x*y))/((np.mean(x)*np.mean(x))-np.mean(x*x)))
        m=round(m)
        return m
    except Exception as e:
        print (e)

x=[5, 1, 4]
y=[0, 10]

print(type(x[1:]))
print(slope(x, y))

g=0
f=g
g=1+2
if(1==1):
    continue
else:
    print("ana are mere")
print(f)'''


'''            if len(esd_axis)>1:
                current_slope_value=slope(time_axis[len(time_axis)-2:], esd_axis[len(esd_axis)-2:])
                if len(esd_axis)==2:
                    slope_value=current_slope_value
                    if current_slope_value>0:
                        flag_cps=True
                        sps+=current_slope_value
                    else:
                        flag_cns=True
                        sns+=current_slope_value
                elif current_slope_value>0 and flag_cps is True:
                    sps+=current_slope_value
                    print("1aici", current_slope_value)
                elif current_slope_value>0 and flag_cps is False:
                    sps=current_slope_value
                    flag_cps=True
                    print("2aici", current_slope_value)
                elif current_slope_value<0 and flag_cns is True:
                    sns+=current_slope_value
                    print("3aici", current_slope_value)
                elif current_slope_value<0 and flag_cns is False:
                    sns=current_slope_value
                    flag_cns=True
                    print("4aici", current_slope_value)
            print(sps, sns)'''