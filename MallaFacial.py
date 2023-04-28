import cv2
import mediapipe as mp 
import math

#Input Cam
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#Creamos función de dibujo
mpDraw = mp.solutions.drawing_utils
drawSettings = mpDraw.DrawingSpec(thickness=1, circle_radius= 1)

#Almacenando la malla facial
mpFacialMesh = mp.solutions.face_mesh
facialMesh = mpFacialMesh.FaceMesh(max_num_faces= 2)

#Main
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
            mpDraw.draw_landmarks(frame, faces, mpFacialMesh.FACEMESH_CONTOURS, drawSettings, drawSettings)

            for id, points in enumerate(faces.landmark):
                height, width, c = frame.shape
                x, y = int(points.x * width), int(points.y * height)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])

                if len(lista) == 468:
                    #Ceja derecha
                    x1, y1, = lista[65][1:]
                    x2, y2, = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    longitudCejaDerecha = math.hypot(x2 - x1, y2 - y1)

                    #Ceja Izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2

                    longitudCejaIzquierda = math.hypot(x4 - x3, y4 - y3)

                    if(longitudCejaDerecha < 19 and longitudCejaIzquierda < 19):
                        cv2.putText(frame, "Enojado", (480,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    
    cv2.imshow("Reconocimiento Emociones", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
