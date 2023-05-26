import cv2
import mediapipe as mp
import math

class EmotionRecognition:
    def __init__(self):
        # Constantes para la triangulación
        self.focal_length = 875  # Valor aproximado para la cámara de la laptop
        self.known_distance = 50  # Distancia en cm de la persona a la cámara para calibración
        self.known_width = 16  # Ancho en cm del rostro de la persona para calibración
        self.face_points = [33, 133, 362, 263]  # Puntos clave de la cara (nariz, barbilla, ojos)

        # Inicializar la cámara
        self.res = [1280, 720]
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.res[0])
        self.cap.set(4, self.res[1])

        # Inicializar el reconocimiento facial
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSettings = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.mpFacialMesh = mp.solutions.face_mesh
        self.facialMesh = self.mpFacialMesh.FaceMesh(max_num_faces=2)

    def distance_calculator(self, x1, x2, y1, y2):
        distance = math.hypot(x2 - x1, y2 - y1)
        return distance

    def run(self):
        while True:
            ret, frame = self.cap.read()
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.facialMesh.process(frameRGB)
            lista = []

            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(frame, face, self.mpFacialMesh.FACEMESH_CONTOURS, self.drawSettings,
                                               self.drawSettings)

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

                    if len(lista) == 468 and distance < 82:
                        # Felicidad

                        # Boca extremos
                        x1, y1 = lista[308][1:]
                        x2, y2 = lista[61][1:]
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        distanciaBocaExtremo = self.distance_calculator(x1, x2, y1, y2)

                        # Boca apertura
                        x3, y3 = lista[13][1:]
                        x4, y4 = lista[14][1:]
                        cv2.line(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                        distanciaBocaApertura = self.distance_calculator(x3, x4, y3, y4)

                        # Extremo ojo der - Ceja Derecha
                        x5, y5 = lista[65][1:]
                        x6, y6 = lista[158][1:]
                        cv2.line(frame, (x5, y5), (x6, y6), (0, 0, 255), 3)
                        distanciaCejaDer = self.distance_calculator(x5, x6, y5, y6)

                        # Extremo ojo izq - Ceja Izq
                        x7, y7 = lista[295][1:]
                        x8, y8 = lista[385][1:]
                        cv2.line(frame, (x7, y7), (x8, y8), (0, 0, 255), 3)
                        distanciaCejaIzq = self.distance_calculator(x7, x8, y7, y8)

                        distanciaCejas = (distanciaCejaDer + distanciaCejaIzq) / 2

                        # Mejilla Izquierda
                        x9, y9 = lista[410][1:]
                        x10, y10 = lista[427][1:]
                        cv2.line(frame, (x9, y9), (x10, y10), (0, 0, 255), 3)
                        distainciaMejillaIzq = self.distance_calculator(x9, x10, y9, y10)

                        # Mejilla Der
                        x11, y11 = lista[186][1:]
                        x12, y12 = lista[207][1:]
                        cv2.line(frame, (x11, y11), (x12, y12), (0, 0, 255), 3)
                        distainciaMejillaDer = self.distance_calculator(x11, x12, y11, y12)

                        distanciaMejillas = (distainciaMejillaDer + distainciaMejillaIzq) / 2

                        # Pata Gallo Izq
                        x13, y13 = lista[346][1:]
                        x14, y14 = lista[340][1:]
                        cv2.line(frame, (x13, y13), (x14, y14), (0, 0, 255), 3)
                        distanciaPataGalloIzq = self.distance_calculator(x13, x14, y13, y14)

                        # Pata Gallo Der
                        x15, y15 = lista[117][1:]
                        x16, y16 = lista[111][1:]
                        cv2.line(frame, (x15, y15), (x16, y16), (0, 0, 255), 3)
                        distanciaPataGalloDer = self.distance_calculator(x15, x16, y15, y16)

                        distanciaPataGallo = (distanciaPataGalloDer + distanciaPataGalloIzq) / 2

                        # Pata Gallo Izq 2
                        x17, y17 = lista[448][1:]
                        x18, y18 = lista[342][1:]
                        cv2.line(frame, (x17, y17), (x18, y18), (0, 0, 255), 3)
                        distanciaPataGalloIzq2 = self.distance_calculator(x17, x18, y17, y18)

                        # Pata Gallo Der 2
                        x19, y19 = lista[228][1:]
                        x20, y20 = lista[113][1:]
                        cv2.line(frame, (x19, y19), (x20, y20), (0, 0, 255), 3)
                        distanciaPataGalloDer2 = self.distance_calculator(x19, x20, y19, y20)

                        distanciaPataGallo2 = (distanciaPataGalloIzq2 + distanciaPataGalloDer2) / 2

                        if (
                                distanciaBocaExtremo > 120.0 and
                                distanciaBocaApertura > 23.0 and
                                distanciaCejas > 30.0 and
                                distanciaMejillas < 34.0 and
                                distanciaPataGallo < 12.0 and
                                distanciaPataGallo2 < 35.0):
                            emotion = "Feliz"
                            cv2.putText(frame, str(emotion), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)

                        elif (
                                distanciaCejas < 20.0):
                            emotion = "Enojado"
                            cv2.putText(frame, str(emotion), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)
                        elif (
                            distanciaBocaApertura > 30.0
                        ):
                            emotion = "Sorprendido"
                            cv2.putText(frame, str(emotion), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)
                        

                if distance > 82:
                    cv2.putText(frame, "No se puede detectar", (480, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)

            cv2.imshow("Reconocimiento Emociones", frame)
            k = cv2.waitKey(1)

            if k == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()



