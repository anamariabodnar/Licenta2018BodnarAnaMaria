import cv2
import dlib
import numpy as np
import time
import os
import winsound
from scipy.spatial import distance
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt


fps=27.533868886838793 #constanta care depinde de camera folosita. calculata in scriptul frames_per_seconds.py
total_frames_60sec=fps*60

count_blinks=0
count_close=0
count_frame=0
ear_thresh=0.353424
blinks=0
sps=0 #sum of consecutive positive slope values
sns=0 #sum of consecutive negative slope values

current_slope_value=0
prev_slope_value=current_slope_value
pathOut="frames6"

perclose=48
predictor_path="C:\\Users\\User\\Anaconda3\\Lib\\site-packages\\dlib\\shape_predictor_68_face_landmarks.dat"


predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(m_start, m_end)= face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#cap=cv2.VideoCapture('D:\\an3\\s2\\licenta\\blinks\\blink10.mp4')
cap=cv2.VideoCapture(0)
print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
def threshold(t, image):
    intensity_matrix=np.zeros((len(image), len(image[0]), len(image[0][0])))
    for w in range(0,len(image)):
        for h in range(0,len(image[0])):
            for j in range(0, len(image[0][0])):
                intensity = image[w][h][j]
                if (intensity <= t):
                    x = 0
                else:
                    x = 255
                intensity_matrix[w][h][j]=int(x)
    return np.array(intensity_matrix)

'''def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    window = [
        (i, j)
        for i in range(-indexer, filter_size-indexer)
        for j in range(-indexer, filter_size-indexer)
    ]
    index = len(window) // 2
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = sorted(
                0 if (
                    min(i+a, j+b) < 0
                    or len(data) <= i+a
                    or len(data[0]) <= j+b
                ) else data[i+a][j+b]
                for a, b in window
            )[index]
    return data'''


def median_filter2d(data, filter_size):
    temp=[]
    indexer=filter_size//2
    data_final=np.zeros((len(data), len(data[0])))
    for i in range(0, len(data)):
        for j in range(0, len(data[0])):
            for k in range(0, filter_size):
                if i+k-indexer<0 or i+k-indexer>len(data)-1:
                    for l in range(0, filter_size):
                        temp.append(0)
                else:
                    if j+k-indexer<0 or j+indexer>len(data[0])-1:
                        temp.append(0)
                    else:
                        for z in range(filter_size):
                            temp.append(data[i+k-indexer][j+k-indexer])
            temp.sort()
            data_final[i][j]=temp[len(temp)//2]
            temp=[]
    return data_final

def median_filter3d(data_3d, t):
    data_final=[]
    for i in range (0, len(data_3d)):
        m=median_filter2d(data_3d[i], t)
        data_final.append(m)
    return np.array(data_final)


def search_esd_value(image):
    t=0
    found = False
    while not found:
        #thresh = threshold(t,image)
        rec, thresh=cv2.threshold(image, t ,255, cv2.THRESH_BINARY)
        #median_blur = median_filter3d(thresh, 3)
        median_blur=cv2.medianBlur(thresh, 3)
        #l = [np.unique(ll) for ll in median_blur]
        dmedian_blur = median_blur.ravel()
        sorted_v=np.sort(dmedian_blur)
        #x = np.searchsorted(sorted_v, 0)
        if sorted_v[0] == 0:
            found = True
        else:
            t += 1

    return t
    '''plt.subplot(121), plt.imshow(image), plt.title('BlurredByMe')
        plt.xticks([]), plt.yticks([])
        plt.show() #aici
        #print(median_blur)
        #print(np.searchsorted(median_blur, 0))
        for i in range(0, len(median_blur)):
            for j in range(0, len(median_blur[0])):
                for k in range(0, len(median_blur[0][0])):
                    if median_blur[i][j][k] == 0:
                        found = True
                        ESD_value = t
                        return ESD_value
        t+=1'''


def slope(x_val, y_val):
    x=np.array(x_val)
    y=np.array(y_val)
    m=(((np.mean(x)*np.mean(y))-np.mean(x*y))/((np.mean(x)*np.mean(x))-np.mean(x*x)))
    m=round(m)
    return m

def detect_eye_blinks(slope, sps, sns):
    threshold_value=99
    t_s=time.time()
    flag_close=False
    flag_open=False
    flag_blink=False
    flag_3=False
    s=''
    if sps<=threshold_value:
        flag_open=True
        s="stare 1 (open)"
    if sps > threshold_value or sns <(threshold_value*-1):
        s="stare 2 (close)"
        flag_close=True
    if slope <=0 and flag_close is True:
        s="stare 3 (opening or steady state)"
        flag_3=True
    elif sns <(threshold_value*-1) and flag_3 is True:
        flag_blink=True
        s="stare 4 (blink)"

    '''if flag_blink is True:
        return " "
    if flag_close is True:
        return " "
    if flag_open is True:'''
    if flag_blink is True:
        print("aiciiiiii")
        return s
    else:
        return s


def ear(eye):
    a= np.linalg.norm(eye[1] - eye[5])
    b= np.linalg.norm(eye[2] - eye[4])
    c= np.linalg.norm(eye[0] - eye[3])
    r=(a+b)/(2.0*c)
    return r

def mar(mouth):
    a = np.linalg.norm(mouth[2] - mouth[10])
    b = np.linalg.norm(mouth[4] - mouth[8])
    c = np.linalg.norm(mouth[0] - mouth[6])
    r = (a + b) / (2.0 * c)
    return r


t_start=time.time()
esd_axis=list()
time_axis=list()
mar_values=list()
s1=""
s2=""
flag_start_blink=False
flag_stop_blink=False

while cap.isOpened():
    src, frame = cap.read()
    if src==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces=detector(gray, 0)
        for rect in faces:
            shape=predictor(gray, rect)
            shape=face_utils.shape_to_np(shape)

            left_eye=shape[l_start:l_end]
            right_eye=shape[r_start:r_end]

            mouth=shape[m_start:m_end]
            mouth_ear=mar(mouth)
            mar_values.append(mouth_ear*10)

            l = np.array(left_eye)
            r = np.array(right_eye)

            clone = frame.copy()

            left_eye_ellipse=cv2.fitEllipse(left_eye)
            right_eye_ellipse=cv2.fitEllipse(right_eye)

            mouth_ellipse=cv2.fitEllipse(mouth)

            '''cv2.ellipse(frame, left_eye_ellipse, (0, 255, 0), 2)
            cv2.ellipse(frame, right_eye_ellipse, (0, 255, 0), 2)
            cv2.ellipse(frame, mouth_ellipse, (0, 255, 0), 2)'''

            #for(x, y) in shape:
            #cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            (x, y, w, h) = cv2.boundingRect(np.array([shape[l_start:l_end]]))
            (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[r_start:r_end]]))
            (x2, y2, w2, h2)=cv2.boundingRect(np.array([shape[m_start:m_end]]))
            roi_l = clone[y:y + h, x:x + w]
            roi_r=clone[y1:y1 + h1, x1:x1 + w1]

            t_l = search_esd_value(roi_l)
            t_r=search_esd_value(roi_r)
            t=(t_l+t_r)/2

            esd_axis.append(t)
            time_axis.append(time.time()-t_start)
            #print(t)
            if len(esd_axis)>1:
                current_slope_value=slope(time_axis[len(time_axis)-2:], esd_axis[len(esd_axis)-2:])
                #print("slope here:",current_slope_value)
                if len(esd_axis)==2:
                    prev_slope_value=current_slope_value
                    if current_slope_value>=0:
                        sps+=current_slope_value
                        sns=0
                    else:
                        sns+=current_slope_value
                        sps=0
                elif current_slope_value >= 0 and prev_slope_value >= 0:
                    sps+=current_slope_value
                    sns=0
                    prev_slope_value=current_slope_value
                elif current_slope_value >= 0 and prev_slope_value <= 0:
                    sps=current_slope_value
                    sns=0
                    prev_slope_value=current_slope_value
                elif current_slope_value<0 and prev_slope_value <= 0:
                    sns+=current_slope_value
                    sps=0
                    prev_slope_value=current_slope_value
                elif current_slope_value<0 and prev_slope_value>=0:
                    sns=current_slope_value
                    sps=0
                    prev_slope_value=current_slope_value
            flag_close = False
            flag_open = False
            flag_blink = False
            flag_3 = False
            threshold_value=100
            s = ''
            if sps <= threshold_value:
                flag_open = True
                s = "stare 1 (open)"
            if sps > threshold_value or sns < (threshold_value * -1):
                s = "stare 2 (close)"
                flag_close = True
            if sns < (threshold_value * -1) and flag_close is True:
                s = "stare 3 (blink)"
                flag_3 = True
            elif sns < (threshold_value * -1) and flag_3 is True:
                flag_blink = True
                s = "stare 4 (blink)"
            #print(current_slope_value, sps, sns)


            '''cv2.putText(frame, "esd: "+str(esd_axis[count_frame]), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "slope: "+str(current_slope_value), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imwrite(pathOut + "/%#05d.jpg" % (count_frame), frame)
            count_frame+=1
            cv2.putText(frame, "close", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)'''
            #print(detect_eye_blinks(current_slope_value, sps, sns))
            #s=detect_eye_blinks(current_slope_value, sps, sns)

            if (s == "stare 1 (open)" and s1=="stare 2 (close)" and s2=="stare 3 (blink)") or (s=="stare 1 (open)" and s1=="stare 3 (blink)") or (s=="stare 1 (open)" and s1=="stare 2 (close)" and s2=="stare 1 (open)"):
                count_blinks+=1
                if count_blinks%2==1:
                    flag_start_blink=True
                    flag_stop_blink=False
                else:
                    flag_stop_blink=True
                    flag_start_blink=False
                    cv2.putText(frame, "blinks: " + str(count_blinks // 2), (500, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    blinks=count_blinks//2


            if flag_start_blink is True:
                count_close+=1
                print(count_close)
                if count_close>48:
                    cv2.putText(frame, "ATENTIEEE", (500, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if flag_stop_blink is True:
                count_close=0
            s1=s
            s2=s1

            cv2.putText(frame, s, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "esd: " + str(esd_axis[count_frame]), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "slope: " + str(current_slope_value), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "sps: " + str(sps), (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "sns: " + str(sns), (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "blinks: " + str(count_blinks // 2), (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imwrite(pathOut + "/%#05d.jpg" % (count_frame), frame)
            count_frame += 1
            cv2.putText(frame, "close", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #cv2.putText(frame, str(count_blinks//2), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break


#plt.plot(time_axis,esd_axis, 'b', time_axis, mar_values, 'r')
plt.plot(time_axis,esd_axis, 'b')

plt.xlabel('time', fontsize=12)
plt.ylabel('esd_value', fontsize=12)
#print(len(time_axis), len(esd_axis))
for i,j in zip(time_axis,esd_axis):
    plt.text(i,j, str(j))

'''for i,j in zip(time_axis, mar_values):
    plt.text(i,j, str(round(j,2)))'''
#plt.plot(time_axis, esd_axis)
plt.show()

cap.release()
cv2.destroyAllWindows()