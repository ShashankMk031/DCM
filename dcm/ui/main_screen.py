from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.properties import ObjectProperty
from kivymd.uix.boxlayout import MDBoxLayout

class MainScreen(Screen):
    """Main screen with navigation and content area"""
    screen_manager = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'main_screen'
    
    def switch_screen(self, screen_name):
        """Switch to the specified screen and update button states"""
        print(f"Switching to screen: {screen_name}")  # Debug print
        
        # Switch the screen
        if hasattr(self, 'ids') and 'screen_manager' in self.ids:
            self.ids.screen_manager.current = screen_name
        
        # Update button states if they exist
        nav_buttons = {
            'playlists_btn': 'playlists_screen',
            'library_btn': 'library_screen',
            'generate_btn_nav': 'generate_screen'
        }
        
        for btn_id, target_screen in nav_buttons.items():
            if hasattr(self.ids, btn_id):
                btn = self.ids[btn_id]
                # Make the active button slightly darker
                btn.md_bg_color = (0.1, 0.5, 0.8, 1) if target_screen == screen_name else (0.13, 0.59, 0.95, 1)

class ContentNavigationDrawer(MDBoxLayout):
    """Navigation drawer content"""
    nav_drawer = ObjectProperty()