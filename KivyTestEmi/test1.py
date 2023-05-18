
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class a123(BoxLayout):
    pass

class TutorialApp(App):
    def build(self):
        return a123()

if __name__ == "__main__":
    TutorialApp().run()