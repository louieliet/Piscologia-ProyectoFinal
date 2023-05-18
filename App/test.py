
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
#---------Liz
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder
from kivy.uix.button import Button

class Ui(ScreenManager):
    pass

class MainApp(MDApp):
    def build(self):
        #self.theme_cls.theme_style='Dark'
        #self.theme_cls.primary_palette='Teal'
        button=Button (background_normal='botones.png')
        Builder.load_file('tutorial.kv')
        return Ui()
    def change_style(self,checked,value):
        if value:
            self.theme_cls.theme_style='Dark'
        else:
            self.theme_cls.theme_style='Ligth'




if __name__ == "__main__":
    MainApp().run()