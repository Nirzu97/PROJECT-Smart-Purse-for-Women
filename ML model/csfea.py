from kivy.app import App
from KivyCalendar import DatePicker

class MyApp(App):

    def build(self):
        return DatePicker()
 
MyApp().run()
