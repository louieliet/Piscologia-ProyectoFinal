
import cv2
import mediapipe as mp 
import math
import numpy as np

class facialDetection():
    def __init__(self):
        self.res = [1280, 720]
        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, self.res[0])
        self.cap.set(4, self.res[1])

        #Creamos función de dibujo
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSettings = self.mpDraw.DrawingSpec(thickness=1, circle_radius= 1)

        #Almacenando la malla facial
        self.mpFacialMesh = mp.solutions.face_mesh
        self.facialMesh = self.mpFacialMesh.FaceMesh(max_num_faces= 1)

        # Constantes para la triangulación
        self.focal_length = 875 # Valor aproximado para la cámara de la laptop
        self.known_distance = 50 # Distancia en cm de la persona a la cámara para calibración
        self.known_width = 16 # Ancho en cm del rostro de la persona para calibración
        self.face_points = [33, 133, 362, 263] # Puntos clave de la cara (nariz, barbilla, ojos)

    def distanceCalculator(self, x1, x2, y1, y2):
        distance = math.hypot(x2 - x1, y2 - y1)
        return distance

    def evaluate_emotions(self, emotions, distance):
        distancia_referencia = 90
        distancia_actual = distance
        relacion_escala = distancia_actual / distancia_referencia
        distancias_normalizadas = [distancia * relacion_escala for distancia in emotions]
        suma = 0
        for numero in distancias_normalizadas:
            suma += numero
        promedio = suma / len(distancias_normalizadas)
        return promedio

    
    def facialDetectionStart(self):
        while True:
            ret, frame = self.cap.read()
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.facialMesh.process(frameRGB)

            lista = []

            if results.multi_face_landmarks:

                for face in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(frame, face, self.mpFacialMesh.FACEMESH_CONTOURS, self.drawSettings, self.drawSettings)

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
                        distance = (self.known_width * self.focal_length) / pixel_width
                        cv2.putText(frame, f"{distance:.2f} cm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    for id, points in enumerate(face.landmark):
                        xo, yo = int(points.x * width), int(points.y * height)
                        lista.append([id, xo, yo])
                        cv2.putText(frame, str(id), (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

                        if len(lista) == 468 and distance < 90 and distance > 80:

                            emotions = []
                            emotion = ''

                            # Felicidad

                            # Boca extremos
                            x1, y1 = lista[308][1:]
                            x2, y2 = lista[61][1:]
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            distanciaBocaExtremo = self.distanceCalculator(x1, x2, y1, y2)

                            # Boca apertura
                            x3, y3 = lista[13][1:]
                            x4, y4 = lista[14][1:]
                            cv2.line(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                            distanciaBocaApertura = self.distanceCalculator(x3, x4, y3, y4)

                            # Extremo ojo der - Ceja Derecha
                            x5, y5 = lista[65][1:]
                            x6, y6 = lista[158][1:]
                            cv2.line(frame, (x5, y5), (x6, y6), (0, 0, 255), 3)
                            distanciaCejaDer = self.distanceCalculator(x5, x6, y5, y6)

                            # Extremo ojo izq - Ceja Izq
                            x7, y7 = lista[295][1:]
                            x8, y8 = lista[385][1:]
                            cv2.line(frame, (x7, y7), (x8, y8), (0, 0, 255), 3)
                            distanciaCejaIzq = self.distanceCalculator(x7, x8, y7, y8)

                            distanciaCejas = (distanciaCejaDer + distanciaCejaIzq) / 2

                            # Mejilla Izquierda
                            x9, y9 = lista[410][1:]
                            x10, y10 = lista[427][1:]
                            cv2.line(frame, (x9, y9), (x10, y10), (0, 0, 255), 3)
                            distainciaMejillaIzq = self.distanceCalculator(x9, x10, y9, y10)

                            # Mejilla Der
                            x11, y11 = lista[186][1:]
                            x12, y12 = lista[207][1:]
                            cv2.line(frame, (x11, y11), (x12, y12), (0, 0, 255), 3)
                            distainciaMejillaDer = self.distanceCalculator(x11, x12, y11, y12)

                            distanciaMejillas = (distainciaMejillaDer + distainciaMejillaIzq) / 2

                            # Pata Gallo Izq
                            x13, y13 = lista[346][1:]
                            x14, y14 = lista[340][1:]
                            cv2.line(frame, (x13, y13), (x14, y14), (0, 0, 255), 3)
                            distanciaPataGalloIzq = self.distanceCalculator(x13, x14, y13, y14)

                            # Pata Gallo Der
                            x15, y15 = lista[117][1:]
                            x16, y16 = lista[111][1:]
                            cv2.line(frame, (x15, y15), (x16, y16), (0, 0, 255), 3)
                            distanciaPataGalloDer = self.distanceCalculator(x15, x16, y15, y16)

                            distanciaPataGallo = (distanciaPataGalloDer + distanciaPataGalloIzq) / 2

                            # Pata Gallo Izq 2
                            x17, y17 = lista[448][1:]
                            x18, y18 = lista[342][1:]
                            cv2.line(frame, (x17, y17), (x18, y18), (0, 0, 255), 3)
                            distanciaPataGalloIzq2 = self.distanceCalculator(x17, x18, y17, y18)

                            # Pata Gallo Der 2
                            x19, y19 = lista[228][1:]
                            x20, y20 = lista[113][1:]
                            cv2.line(frame, (x19, y19), (x20, y20), (0, 0, 255), 3)
                            distanciaPataGalloDer2 = self.distanceCalculator(x19, x20, y19, y20)

                            distanciaPataGallo2 = (distanciaPataGalloIzq2 + distanciaPataGalloDer2) / 2

                            emotions.append(distanciaBocaExtremo)
                            emotions.append(distanciaBocaApertura)
                            emotions.append(distanciaCejas)
                            emotions.append(distanciaMejillas)
                            emotions.append(distanciaPataGallo)
                            emotions.append(distanciaPataGallo2)

                            self.promedio = self.evaluate_emotions(emotions, distance)

                            if self.promedio > 30 and self.promedio < 31.7:
                                emotion = "Normal"
                            elif self.promedio > 32 and self.promedio < 35:
                                emotion = "Disgusto"
                            elif self.promedio > 36 and self.promedio < 40:
                                emotion = "Feliz"
                            elif self.promedio > 40 and self.promedio < 44:
                                emotion = "Sorprendido"
                                
                            cv2.putText(frame, str(emotion), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,0), 10)

                        if distance >= 90:
                            cv2.putText(frame, "No se puede detectar", (480, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)
                            
                            
            cv2.imshow("Reconocimiento Emociones", frame)
            k = cv2.waitKey(1)

            if k == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


main = facialDetection()
main.facialDetectionStart()