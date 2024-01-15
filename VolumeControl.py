import numpy as numpy
import cv2
import mediapipe as mediapipe
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

capture = cv2.VideoCapture(0)
mpHands = mediapipe.solutions.hands
hands = mpHands.Hands()
mpDraw = mediapipe.solutions.drawing_utils

device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

while True:
    success, image = capture.read()
    RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finish = hands.process(RGBimage)
    lmList = []
    if finish.multi_hand_landmarks:
        for handlandmark in finish.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                height, width, channel = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append([id, cx, cy])
                mpDraw.draw_landmarks(image, handlandmark, mpHands.HAND_CONNECTIONS)
    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(image, (x1, y1), 5, (255, 255, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 5, (255, 255, 255), cv2.FILLED)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        length = hypot(x2-x1, y2-y1)
        vol = numpy.interp(length, [10, 210], [volMin, volMax])

        volume.SetMasterVolumeLevel(vol, None)
        cv2.putText(image, "{}%".format(round(volume.GetMasterVolumeLevelScalar()*100)),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_4)
    cv2.imshow("Image", image)
    cv2.waitKey(1)
