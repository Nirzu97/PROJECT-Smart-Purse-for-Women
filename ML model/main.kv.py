# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:12:11 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:02:43 2019

@author: Dell
"""

ScreenManagement:
    transition: FadeTransition()
    MainScreen:
    AnotherScreen:
	
<MainScreen>:
    name: 'main'

    Button:
        on_release: app.root.current = 'other'
        text: 'Another Screen'
        font_size: 50
            
<AnotherScreen>:
    name: 'other'

    Button:
        on_release: app.root.current = 'main'
        text: 'back to the home screen'
        font_size: 50