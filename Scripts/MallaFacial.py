import cv2
import mediapipe as mp 
import math
import numpy as np

def distanceCalculator(x1, x2, y1, y2):
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance

#Input Cam
res = [1280, 720]
cap = cv2.VideoCapture(1)
cap.set(3, res[0])
cap.set(4, res[1])


#Creamos función de dibujo
mpDraw = mp.solutions.drawing_utils
drawSettings = mpDraw.DrawingSpec(thickness=1, circle_radius= 1)

#Almacenando la malla facial
mpFacialMesh = mp.solutions.face_mesh
facialMesh = mpFacialMesh.FaceMesh(max_num_faces= 2)

# Constantes para la triangulación
focal_length = 875 # Valor aproximado para la cámara de la laptop
known_distance = 50 # Distancia en cm de la persona a la cámara para calibración
known_width = 16 # Ancho en cm del rostro de la persona para calibración
face_points = [33, 133, 362, 263] # Puntos clave de la cara (nariz, barbilla, ojos)

#Main
while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facialMesh.process(frameRGB)

    lista = []

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, face, mpFacialMesh.FACEMESH_CONTOURS, drawSettings, drawSettings)

            height, width, c = frame.shape

            face_points = [33, 133, 362, 263]

            points = []
            for p in face_points:
                landmark = face.landmark[p]
                x, y, z = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]), landmark.z
                points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            if len(points) == len(face_points):
                pixel_width = abs(points[0][0] - points[3][0])
                distance = (known_width * focal_length) / pixel_width
                #print(distance)
                cv2.putText(frame, f"{distance:.2f} cm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            for id, points in enumerate(face.landmark):
            
                xo, yo = int(points.x * width), int(points.y * height)
                lista.append([id, xo, yo])

        
                #cv2.putText(frame, str(id), (xo,yo), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

                if len(lista) == 468 and distance < 278.76:

                    #Felicidad
                    
                    #Mejilla izq - Ojo izq
                    x, y = lista[266][1:]
                    x1, y1 = lista[374][1:]
                    d1 = distanceCalculator(x, x1, y, y1)
                    #cv2.line(frame, (x,y), (x1,y1), (0,0,0), 3)

                    #Boca apertura 
                    x2, y2 = lista[185][1:]
                    x3, y3 = lista[409][1:]
                    d2 = distanceCalculator(x2, x3, y2, y3)
                    #cv2.line(frame, (x2,y2), (x3,y3), (0,0,0), 3)
                    

                    #Mejilla der - Ojo der
                    x4, y4 = lista[36][1:]
                    x5, y5 = lista[145][1:]
                    d3 = distanceCalculator(x4, x5, y4, y5)
                    #cv2.line(frame, (x4,y4), (x5,y5), (0,0,0), 3)


                    #Ojera izq - Ojo izq
                    x6, y6 = lista[449][1:]
                    x7, y7 = lista[373][1:]
                    d4 = distanceCalculator(x6, x7, y6, y7)
                    #cv2.line(frame, (x6,y6), (x7,y7), (0,0,0), 3)

                    #Ojera der - Ojo der
                    x8, y8 = lista[229][1:]
                    x9, y9 = lista[144][1:]
                    d5 = distanceCalculator(x8, x9, y8, y9)

                    promedio = (d1 + d2 + d3 + d4 + d5) / 5

                    print(promedio)

                    if (promedio < 22 and promedio > 20):
                        cv2.putText(frame, "Feliz", (480, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)
                
                if distance > 218.76:
                    cv2.putText(frame, "No se puede detectar", (480, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)
                    
                    
    cv2.imshow("Reconocimiento Emociones", frame)
    k = cv2.waitKey(1)

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()




