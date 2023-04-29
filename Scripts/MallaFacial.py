import cv2
import mediapipe as mp 
import math

#Input Cam
res = [1280, 720]
cap = cv2.VideoCapture(0)
cap.set(3,res[0])
cap.set(4,res[1])

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

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, face, mpFacialMesh.FACEMESH_CONTOURS, drawSettings, drawSettings)

            height, width, c = frame.shape

            #x = int(face.landmark[1].x * width)
            #y = int(face.landmark[1].y * height)



            for id, points in enumerate(face.landmark):
                x, y = int(points.x * width), int(points.y * height)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])
                
                cv2.putText(frame, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 3)

                if len(lista) == 468:
                    #Ceja derecha
                    x1, y1 = lista[1][1:]




                    #if(longitudCejaDerecha < 15 and longitudCejaIzquierda < 15):
                        #cv2.putText(frame, "No", (480,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

    
    cv2.imshow("Reconocimiento Emociones", frame)
    k = cv2.waitKey(1)

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
