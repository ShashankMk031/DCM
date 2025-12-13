"""
LoadingDialog - A customizable loading dialog with progress tracking.

This module provides a LoadingDialog class that can be used to show progress
of long-running operations with detailed status updates.
"""

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivymd.uix.dialog import MDDialog
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.button import MDFlatButton


class LoadingDialog(MDDialog):
    """A customizable loading dialog with progress tracking.
    
    Features:
    - Progress bar with configurable range
    - Title and status message display
    - Cancel button support
    - Auto-dismiss on completion
    """
    
    # Properties
    progress = NumericProperty(0)
    max_progress = NumericProperty(100)
    status_text = StringProperty("Processing...")
    cancel_button_text = StringProperty("CANCEL")
    show_cancel_button = BooleanProperty(True)
    
    def __init__(self, **kwargs):
        """Initialize the loading dialog."""
        # Create content
        self.content_cls = MDBoxLayout(
            orientation='vertical',
            spacing=dp(10),
            padding=dp(15),
            size_hint_y=None,
            height=dp(150)
        )
        
        # Add status label
        self.status_label = MDLabel(
            text=self.status_text,
            halign='left',
            theme_text_color='Secondary',
            font_style='Body1',
            size_hint_y=None,
            height=dp(24)
        )
        self.content_cls.add_widget(self.status_label)
        
        # Add progress bar
        self.progress_bar = MDProgressBar(
            min=0,
            max=self.max_progress,
            value=self.progress,
            size_hint_y=None,
            height=dp(4)
        )
        self.content_cls.add_widget(self.progress_bar)
        
        # Add progress text
        self.progress_label = MDLabel(
            text=self._get_progress_text(),
            halign='right',
            theme_text_color='Secondary',
            font_style='Caption',
            size_hint_y=None,
            height=dp(20)
        )
        self.content_cls.add_widget(self.progress_label)
        
        # Create buttons
        self.cancel_button = MDFlatButton(
            text=self.cancel_button_text,
            theme_text_color="Custom",
            text_color=self.theme_cls.primary_color,
            on_release=self._on_cancel
        )
        
        # Initialize dialog
        super().__init__(
            title="",
            type="custom",
            content_cls=self.content_cls,
            size_hint=(0.8, None),
            height=dp(200),
            buttons=[self.cancel_button] if self.show_cancel_button else [],
            auto_dismiss=False
        )
        
        # Bind properties
        self.bind(progress=self._update_progress)
        self.bind(status_text=self._update_status)
        self.bind(show_cancel_button=self._update_buttons)
        
        # Operation state
        self._is_cancelled = False
        self._on_cancel_callback = None
    
    def _get_progress_text(self) -> str:
        """Get formatted progress text."""
        if self.max_progress > 0:
            return f"{int(self.progress)} of {int(self.max_progress)}"
        return f"{int(self.progress)}%"
    
    def _update_progress(self, instance, value):
        """Update progress bar and text when progress changes."""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.value = value
            self.progress_label.text = self._get_progress_text()
            
            # Auto-dismiss if max progress reached
            if self.max_progress > 0 and value >= self.max_progress:
                Clock.schedule_once(lambda dt: self.dismiss(), 0.5)
    
    def _update_status(self, instance, value):
        """Update status text when it changes."""
        if hasattr(self, 'status_label'):
            self.status_label.text = value
    
    def _update_buttons(self, instance, value):
        """Update visibility of cancel button."""
        if hasattr(self, 'cancel_button'):
            if value and self.cancel_button not in self.buttons:
                self.buttons.insert(0, self.cancel_button)
            elif not value and self.cancel_button in self.buttons:
                self.buttons.remove(self.cancel_button)
    
    def _on_cancel(self, *args):
        """Handle cancel button press."""
        self._is_cancelled = True
        if self._on_cancel_callback:
            self._on_cancel_callback()
        self.dismiss()
    
    def set_cancel_callback(self, callback):
        """Set a callback to be called when the cancel button is pressed."""
        self._on_cancel_callback = callback
    
    def is_cancelled(self) -> bool:
        """Check if the operation was cancelled."""
        return self._is_cancelled
    
    def update(self, progress: float = None, status: str = None, max_progress: float = None):
        """Update progress and/or status.
        
        Args:
            progress: New progress value
            status: New status text
            max_progress: New maximum progress value
        """
        if progress is not None:
            self.progress = progress
        if status is not None:
            self.status_text = status
        if max_progress is not None:
            self.max_progress = max_progress
    
    def show(self):
        """Show the dialog."""
        self.open()
    
    def close(self):
        """Close the dialog."""
        self.dismiss()
