import cv2
import mediapipe as mp
import math
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.image import Texture
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivymd.uix.textfield import MDTextFieldRect
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.core.window import Window
import numpy as np
from PIL import Image as PILImage
from PIL import ImageFilter
import io
from kivy.graphics.texture import Texture
from kivymd.uix.controllers import WindowController
from kivymd.uix.screen import MDScreen
from kivy.core.window import Window
from kivy.config import Config
from collections import Counter
from kivy.uix.label import Label
import statistics
from kivy.properties import ListProperty
from kivymd.uix.textfield import MDTextField


class EmotionRecognition(Screen):
    def __init__(self, **kwargs):
        super(EmotionRecognition, self).__init__(**kwargs)

        self.emotion_mode = ' '
        self.registered_emotions = []
        # Constantes para la triangulación
        self.focal_length = 875  # Valor aproximado para la cámara de la laptop
        self.known_distance = 50  # Distancia en cm de la persona a la cámara para calibración
        self.known_width = 16  # Ancho en cm del rostro de la persona para calibración
        self.face_points = [33, 133, 362, 263]  # Puntos clave de la cara (nariz, barbilla, ojos)

        self.cap = cv2.VideoCapture(0)

        # Inicializar el reconocimiento facial
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSettings = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.mpFacialMesh = mp.solutions.face_mesh
        self.facialMesh = self.mpFacialMesh.FaceMesh(max_num_faces=1)

        # Crear una instancia de la imagen para mostrar el fotograma capturado
        self.image = Image()
        self.add_widget(self.image)


        self.label = Label(text='Detección de emociones', pos_hint={'x': 0.1, 'top': 0.95}, size_hint=(0.8, 0.05), halign='center', font_name='fonts/DellaRespira-Regular.ttf')
        self.add_widget(self.label)

        # Programar la actualización del fotograma
        Clock.schedule_interval(self.update_frame, 1.0 / 60.0)


    def apply_blur(self, frame):
        blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)  # Ajusta los valores del tamaño del kernel según tus necesidades
        return blurred_frame
    
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

    def update_frame(self, dt):
        ret, frame = self.cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.facialMesh.process(frame_rgb)

        lista = []


        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                #self.mpDraw.draw_landmarks(frame, face, self.mpFacialMesh.FACEMESH_CONTOURS, self.drawSettings, self.drawSettings)

                height, width, c = frame.shape

                face_points = [33, 133, 362, 263]

                points = []
             
                
                for p in face_points:
                    landmark = face.landmark[p]
                    x, y, z = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]), landmark.z
                    points.append((x, y))
                    #cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
      
                if len(points) == len(face_points):
                    pixel_width = abs(points[0][0] - points[3][0])
                    distance = (self.known_width * self.focal_length) / pixel_width
                    #cv2.putText(frame, f"{distance:.2f} cm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                for id, points in enumerate(face.landmark):
                    xo, yo = int(points.x * width), int(points.y * height)
                    lista.append([id, xo, yo])
                    #cv2.putText(frame, str(id), (xo, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

                    if len(lista) == 468 and distance < 104 and distance > 80:

                        self.label.text = 'Detección de emociones'

                        emotions = []
                        emotion = ''

                        # Felicidad

                        # Boca extremos
                        x1, y1 = lista[308][1:]
                        x2, y2 = lista[61][1:]
                        #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        distanciaBocaExtremo = self.distanceCalculator(x1, x2, y1, y2)

                        # Boca apertura
                        x3, y3 = lista[13][1:]
                        x4, y4 = lista[14][1:]
                        #cv2.line(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                        distanciaBocaApertura = self.distanceCalculator(x3, x4, y3, y4)

                        # Extremo ojo der - Ceja Derecha
                        x5, y5 = lista[65][1:]
                        x6, y6 = lista[158][1:]
                        #cv2.line(frame, (x5, y5), (x6, y6), (0, 0, 255), 3)
                        distanciaCejaDer = self.distanceCalculator(x5, x6, y5, y6)

                        # Extremo ojo izq - Ceja Izq
                        x7, y7 = lista[295][1:]
                        x8, y8 = lista[385][1:]
                        #cv2.line(frame, (x7, y7), (x8, y8), (0, 0, 255), 3)
                        distanciaCejaIzq = self.distanceCalculator(x7, x8, y7, y8)

                        distanciaCejas = (distanciaCejaDer + distanciaCejaIzq) / 2

                        # Mejilla Izquierda
                        x9, y9 = lista[410][1:]
                        x10, y10 = lista[427][1:]
                        #cv2.line(frame, (x9, y9), (x10, y10), (0, 0, 255), 3)
                        distainciaMejillaIzq = self.distanceCalculator(x9, x10, y9, y10)

                        # Mejilla Der
                        x11, y11 = lista[186][1:]
                        x12, y12 = lista[207][1:]
                        #cv2.line(frame, (x11, y11), (x12, y12), (0, 0, 255), 3)
                        distainciaMejillaDer = self.distanceCalculator(x11, x12, y11, y12)

                        distanciaMejillas = (distainciaMejillaDer + distainciaMejillaIzq) / 2

                        # Pata Gallo Izq
                        x13, y13 = lista[346][1:]
                        x14, y14 = lista[340][1:]
                        #cv2.line(frame, (x13, y13), (x14, y14), (0, 0, 255), 3)
                        distanciaPataGalloIzq = self.distanceCalculator(x13, x14, y13, y14)

                        # Pata Gallo Der
                        x15, y15 = lista[117][1:]
                        x16, y16 = lista[111][1:]
                        #cv2.line(frame, (x15, y15), (x16, y16), (0, 0, 255), 3)
                        distanciaPataGalloDer = self.distanceCalculator(x15, x16, y15, y16)

                        distanciaPataGallo = (distanciaPataGalloDer + distanciaPataGalloIzq) / 2

                        # Pata Gallo Izq 2
                        x17, y17 = lista[448][1:]
                        x18, y18 = lista[342][1:]
                        #cv2.line(frame, (x17, y17), (x18, y18), (0, 0, 255), 3)
                        distanciaPataGalloIzq2 = self.distanceCalculator(x17, x18, y17, y18)

                        # Pata Gallo Der 2
                        x19, y19 = lista[228][1:]
                        x20, y20 = lista[113][1:]
                        #cv2.line(frame, (x19, y19), (x20, y20), (0, 0, 255), 3)
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
                            self.registered_emotions.append("Normal")
                        elif self.promedio > 32 and self.promedio < 35:
                            emotion = "Disgusto"
                            self.registered_emotions.append("Disgusto")
                        elif self.promedio > 36 and self.promedio < 40:
                            emotion = "Feliz"
                            self.registered_emotions.append("Feliz")
                        elif self.promedio > 40 and self.promedio < 44:
                            emotion = "Sorprendido"
                            self.registered_emotions.append("Sorprendido")
                        
                        #print(self.promedio)
                        
                        #print(emotion)
                    
                    elif distance > 104 or distance < 80:
                        self.label.text = 'No se puede detectar'


        blurred_frame = self.apply_blur(frame)



        # Mostrar el fotograma en la imagen de Kivy
        texture = Texture.create(size=(blurred_frame.shape[1], blurred_frame.shape[0]))
        texture.flip_vertical()
        texture.blit_buffer(blurred_frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture


    def get_emotionmode(self):
        try:
            moda = statistics.mode(self.registered_emotions)
            self.registered_emotions.clear()
            print(moda)
            return moda
        except statistics.StatisticsError:
            # Manejo del error cuando no hay moda para los datos vacíos
            print("No hay modo disponible para los datos vacíos")
            return None

    def on_stop(self):
        self.cap.release()
        super(EmotionRecognition, self).on_stop()

class AppManager(ScreenManager):
    pass

class Cupones(Screen):
    pass
class Inicio(Screen):
    pass

class MiEstilo(Screen):
    pass

class Catalogo(Screen):
    pass

class LoginApp(Screen):
    pass

class Menu(Screen):
    pass
class Cupon1(Screen):
    pass
class Cupon2(Screen):
    pass
class Cupon3(Screen):
    pass
class Cupon4(Screen):
    pass
class TestResult(Screen):
    pass
class PreEmotionRecognition(Screen):
    pass

class EmotionRecognitionApp(MDApp):
    next_button_count = 0
    image_paths = ['Sources and Tools/R1.jpg', 'Sources and Tools/R2.png', 'Sources and Tools/R3.png']
    emotion_data = []
    def build(self):
        # Establecer el tamaño fijo de la ventana
        Window.top = 0
        Window.size = (400, 800)
        Window.borderless = True
        Window.fullscreen = False
        kv = Builder.load_file('tutorial.kv')
        self.theme_cls.theme_style='Dark'
        self.theme_cls.primary_palette='Teal'
        return kv
    
    def sign_in(self):
        username = self.root.get_screen('registerscreen').ids.sign_in_username.text
        password = self.root.get_screen('registerscreen').ids.sign_in_password.text
        # Lógica para el inicio de sesión
        if username == "admin" and password == "123":
            # Iniciar sesión exitoso
            self.root.current = 'menu'
        else:
            # Mostrar mensaje de error
            dialog = MDDialog(
                text="Credenciales inválidas. Inténtalo de nuevo.",
                buttons=[
                    MDFlatButton(text="Aceptar", on_release=lambda x: dialog.dismiss())
                    
                ]
            )
            dialog.open()
    
    def startpassImage(self):
        self.emotion_data.clear()
        screen = self.root.get_screen('EmotionRecognition')

        if self.next_button_count == 0:
            image_path = self.image_paths[0]
            screen.ids.recomendation_image.source = image_path
            
            emotion_mode = screen.get_emotionmode()
            data = {'emotion_mode': emotion_mode, 'image_path': image_path}
            self.emotion_data.append(data)
            self.next_button_count += 1


    def passImage_emotion(self):
        screen = self.root.get_screen('EmotionRecognition')

        if self.next_button_count < len(self.image_paths):
            screen.ids.recomendation_image.source = self.image_paths[self.next_button_count]
            image_path = self.image_paths[self.next_button_count]
            
            emotion_mode = screen.get_emotionmode()
            data = {'emotion_mode': emotion_mode, 'image_path': image_path}
            self.emotion_data.append(data)
            self.next_button_count += 1

        else:
            self.next_button_count = 0  # Reiniciar el contador después de mostrar todas las imágenes
            self.root.current = 'testResult'
    
    def show_emotionResult(self):

        for element in self.emotion_data:
            print(element)

        for element in self.emotion_data:
            if element['emotion_mode'] == 'Feliz' or element['emotion_mode'] == 'Sorprendido':
                screen = self.root.get_screen('testResult')
                screen.ids.testResult_label.text = 'Según nuestro análisis, la recomendación que más te gustó fue: '
                print(element['image_path'])
                screen.ids.recomendation_image.source = str(element['image_path'])
            else:
                screen = self.root.get_screen('testResult')
                screen.ids.testResult_label.text = 'No tenemos suficientes evidencias para determinar cuál te gustó más, pero creemos que fue esta: '


if __name__ == '__main__':
    EmotionRecognitionApp().run()
