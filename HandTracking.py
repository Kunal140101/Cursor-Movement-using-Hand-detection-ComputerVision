from sre_constants import SUCCESS
import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands=mpHands.Hands()
mpDraw= mp.solutions.drawing_utils



while True:
    SUCCESS, img= cap.read()
    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            for id,lm in enumerate(handLms.landmark):
                h,w,c=img.shape
                cx,cy= int(lm.x * w), int(lm.y * h)
                # print(id,cx,cy)
                if(id==8):
                    cv2.circle(img,(cx,cy),12,(250, 250, 51 ),cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    


    cv2.imshow("Image",img)
    cv2.waitKey(1)