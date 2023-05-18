#--------------Inicio------------#
#import kivy
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import builder
from kivy.app import App

class Ui(ScreenManager):
    pass
class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style='Dark'
        self.theme_cls.primary_palette ='Teal'
        builder.load_file('desing.kv')
        return Ui()



if __name__=="__main__":
    MainApp().run()

