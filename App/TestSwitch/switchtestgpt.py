from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp

# Definir las clases de las pantallas
class Screen1(Screen):
    pass

class Screen2(Screen):
    pass

# Definir el administrador de pantallas
class ScreenManagerApp(ScreenManager):
    pass

# Crear el archivo .kv
kv = Builder.load_string('''
ScreenManagerApp:
    Screen1:
    Screen2:

<Screen1>:
    name: 'screen1'
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Pantalla 1'
        Button:
            text: 'Ir a Pantalla 2'
            on_release: app.root.current = 'screen2'

<Screen2>:
    name: 'screen2'
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Pantalla 2'
        Button:
            text: 'Ir a Pantalla 1'
            on_release: app.root.current = 'screen1'
''')

# Definir la aplicación principal
class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style='Dark'
        self.theme_cls.primary_palette='Teal'
        return kv

if __name__ == '__main__':
    MainApp().run()
