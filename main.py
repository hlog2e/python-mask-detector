import numpy as np
import cv2
import tensorflow.keras
from pygame import mixer
from playsound import playsound
import asyncio
import time




# 얼굴인식 데이터 가져오기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 학습된 데이터 가져오기
model_filename ='./keras_model.h5'

# 케라스 모델 가져오기
model = tensorflow.keras.models.load_model(model_filename)

# 웹캠 0번으로 부터 캡쳐시작
capture = cv2.VideoCapture(0)

# 영상의 Width와 Height 크기조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# 이미지 처리하기
def preprocessing(frame):
    # frame_fliped = cv2.flip(frame, 1)
    # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


    # 이미지 정규화
    # astype : 속성
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    # keras 모델에 공급할 올바른 모양의 배열 생성
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    # print(frame_reshaped)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detect_face = "no"
    # 얼굴인식 후 사각형 그리기
    for (x, y, w, h) in faces:
        detect_face = "yes"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


    return frame_reshaped, detect_face

#예측 펑션
def predict(frame):
    prediction = model.predict(frame)
    return prediction

yesCount = 0;
noCount = 0;




while True:
    if cv2.waitKey(10) > 0:
        break

    ret, frame = capture.read()
    preprocessed = preprocessing(frame)[0]
    prediction = predict(preprocessed)

    detect_face = preprocessing(frame)[1]
    if (detect_face == "yes"):
        cv2.putText(frame, "No Mask", (20, 70), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255))
        if noCount == 1:
            playsound('noti.mp3', False)
        if noCount > 25:
            noCount = 0

        noCount = noCount + 1
        print(noCount)

    if (detect_face == "no"):
        if (prediction[0, 0] < prediction[0, 1]):
            print('Yes Mask')
            cv2.putText(frame, "Yes Mask", (20, 70), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0))
            # (넣을영상,넣을텍스트,텍스트위치(x,y),폰트명,폰트크기,색상(R,G,B))
            if yesCount == 1:
                playsound('yes_noti.mp3', False)
            if yesCount > 35:
                yesCount = 0

            yesCount = yesCount + 1
            print(yesCount)


        else:
            cv2.putText(frame, "Face not Detected", (20, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255))
            # (넣을영상,넣을텍스트,텍스트위치(x,y),폰트명,폰트크기,색상(R,G,B))
            print('No Detected')
    print(detect_face)



    cv2.imshow("Mask Detector", frame)
    #영상을 출력한다.