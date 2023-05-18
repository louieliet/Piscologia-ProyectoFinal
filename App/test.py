
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class liz(BoxLayout):
    pass

class TutorialApp(App):
    def build(self):
        return liz()

if __name__ == "__main__":
    TutorialApp().run()