import cv2
import dlib
import numpy
import time
import winsound
from scipy.spatial import distance
from imutils import face_utils
from matplotlib import pyplot as plt

def turn_on_alarm():
    winsound.PlaySound('sound.wav', winsound.SND_FILENAME)

def ear(eye):
    a= numpy.linalg.norm(eye[1] - eye[5])
    b= numpy.linalg.norm(eye[2] - eye[4])
    c= numpy.linalg.norm(eye[0] - eye[3])
    r=(a+b)/(2.0*c)
    return r

#print (ear(e))

fps=27.533868886838793 #constanta care depinde de camera folosita. calculata in scriptul frames_per_seconds.py
'''numar total de frame-uri dintr-o secunda il calculez inmultind numarul de secunde cu framerate
s aam ceva intre camera si sofer care sa faca imiaginea clara//de la cristina
https://www.quora.com/How-many-frames-are-in-a-2-minute-video
'''
total_frames_60sec=fps*60
count_frame=0
pathOut="frames11"

count=0
ear_thresh=0.300424
blinks=0
array_ear=[]

'''o persoana ce are ochii inchisi cel putin 80% din timp intr-un minut(~48 sescunde) este pe cale de a adormi
vezi: https://pdfs.semanticscholar.org/892f/a8cadedc265d41dd4d0680274fc7c9afa536.pdf'''
perclose=48
predictor_path="C:\\Users\\User\\Anaconda3\\Lib\\site-packages\\dlib\\shape_predictor_68_face_landmarks.dat"
#face_cascade_path = 'C:\\Users\\AnaMariaBodnar\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
#eye_cascade_path='C:\\Users\\AnaMariaBodnar\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml'

#face_cascade = cv2.CascadeClassifier(face_cascade_path)
#video_capture = cv2.VideoCapture(0)
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
time_axis=[]
start_time=time.time()
#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('D:\\an3\\s2\\licenta\\blinks\\blink11.mp4')

print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
t_start=time.time()
while True:
    # Capture frame-by-frame
    #src, frame = video_capture.read()
    src, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces=detector(gray, 0)
    for rect in faces:
        print("asuj", rect)
        shape=predictor(gray, rect)
        shape=face_utils.shape_to_np(shape)

        left_eye=shape[l_start:l_end]
        right_eye=shape[r_start:r_end]


        l=numpy.array(left_eye)
        r= numpy.array(right_eye)

        right_eye_ear=ear(r)
        print("aici", right_eye)
        left_eye_ear=ear(l)
        average_ear=(right_eye_ear+left_eye_ear)/2.0
        #print(average_ear)
        array_ear.append(round(average_ear, 2))
        time_axis.append(time.time() - t_start)


        left_eye_ellipse=cv2.fitEllipse(left_eye)
        right_eye_ellipse=cv2.fitEllipse(right_eye)
        #cv2.ellipse(frame, left_eye_ellipse, (0, 255, 0), 2)
        #cv2.ellipse(frame, right_eye_ellipse, (0, 255, 0), 2)

        if average_ear<ear_thresh:
            count+=1

            #print(count)
            '''if count>=perclose:
                #aici pornesc alarma
                print("alarma pornita")
                #turn_on_alarm()
                alarm=True
                cv2.putText(frame, "Wake up!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)'''
        else:
            '''end_time=time.time()
            seconds=end_time-start_time
            if seconds>59:
                print("ceva)'''
            ''''''
            if count >= 3:
                blinks += 1
            count=0
            alarm=False



            cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for(x, y) in shape:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.imwrite(pathOut + "/%#05d.jpg" % (count_frame), frame)
        count_frame += 1

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.plot(time_axis,array_ear, 'b')

plt.xlabel('time', fontsize=12)
plt.ylabel('ear_value', fontsize=12)
#print(len(time_axis), len(esd_axis))
for i,j in zip(time_axis,array_ear):
    plt.text(i,j, str(j))

'''for i,j in zip(time_axis, mar_values):
    plt.text(i,j, str(round(j,2)))'''
#plt.plot(time_axis, esd_axis)
#plt.show()
cap.release()
cv2.destroyAllWindows()