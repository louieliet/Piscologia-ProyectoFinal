
from kivy.app import App
from kivymd.app import MDApp

from kivy.lang import Builder

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button

from kivy.core.window import Window

from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivymd.uix.textfield import MDTextFieldRect

class AppManager(ScreenManager):
    pass

class Recomendacion(Screen):
    pass

class Cupones(Screen):
    pass
class MiEstilo(Screen):
    pass
class Catalogo(Screen):
    pass

class LoginApp(Screen):
    pass


class Menu(Screen):
    pass

class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style='Dark'
        self.theme_cls.primary_palette='Teal'
        #button=Button (background_normal='botones.png')
        kv = Builder.load_file('tutorial.kv')
        return kv
    def change_style(self,checked,value):
        if value:
            self.theme_cls.theme_style='Dark'
        else:
            self.theme_cls.theme_style='Ligth'




if __name__ == "__main__":
    MainApp().run()