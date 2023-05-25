import cv2
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock

class OpenCVProcessor:
    def __init__(self, video_source):
        self.video_capture = cv2.VideoCapture(video_source)
        self.process_frame()

    def process_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Aquí puedes realizar el procesamiento de imagen con OpenCV
            # Por ejemplo, aplicar un filtro o realizar detección de rostros
            processed_frame = frame  # Utilizamos el fotograma sin procesar como ejemplo

            # Convierte el fotograma de OpenCV a un formato compatible con Kivy
            image_texture = self.convert_frame_to_texture(processed_frame)

            # Actualiza la imagen en la ventana de Kivy
            app.image.texture = image_texture

        # Programa la próxima actualización del fotograma
        Clock.schedule_once(lambda dt: self.process_frame(), 1.0 / 30.0)  # 30 FPS

    def convert_frame_to_texture(self, frame):
        buffer = frame.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        return texture

class OpenCVKivyApp(App):
    def build(self):
        self.image = Image()
        return self.image

    def on_start(self):
        video_source = 0  # Puedes cambiarlo al archivo de video deseado
        self.processor = OpenCVProcessor(video_source)

if __name__ == '__main__':
    OpenCVKivyApp().run()
