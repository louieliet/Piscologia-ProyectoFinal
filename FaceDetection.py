import cv2 
import mediapipe as mp
import math
from deepface import DeepFace

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mpDraw = mp.solutions.drawing_utils
drawSettings = mpDraw.DrawingSpec(thickness=1, circle_radius= 1)

mpfacialMesh = mp.solutions.face_mesh
facialMesh = mpfacialMesh.FaceMesh(max_num_faces= 1)

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facialMesh.process(frameRGB)

    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if results.multi_face_landmarks:
        for faces in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faces, mpfacialMesh.FACEMESH_CONTOURS, drawSettings, drawSettings)

            for id, points in enumerate(faces.landmark):
                height, width, c = frame.shape
                x, y = int(points.x * width), int(points.y * height)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])
                if(len(lista) == 468):

                    #Ceja derecha
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    longitudcR = math.hypot(x2 - x1, y2 - y1)

                    #Ceja izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitudcL = math.hypot(x4 - x3, y4 - y3)

                    #Boca extremos
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx2, cy2 = (x5 + x6) // 2, (y5 + y6) // 2
                    longitudbE = math.hypot(x6 - x5, y6 - y5)

                    #Boca apertura
                    x7, y7 = lista[295][1:]
                    x8, y8 = lista[385][1:]
                    cx2, cy2 = (x7 + x8) // 2, (y7 + y8) // 2
                    longitudbA = math.hypot(x8 - x7, y8 - y7)

                    #Clasificación
                    
                    #Enojado
                    if(longitudcR < 19 and longitudcL < 19 and longitudbE > 80 and longitudbE < 95 and longitudbA < 5):
                        cv2.putText(frame, "Enojado", (480,80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0,0,255), 3)


    cv2.imshow("Reconocimiento de Emociones", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()

                                    


