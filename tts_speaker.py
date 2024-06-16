# Import the required module for text to speech conversion
from gtts import gTTS
# Import pygame for playing the converted audio
import pygame

def speak_text(mytext):
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("welcome.mp3")
    
    # Initialize the mixer module
    pygame.mixer.init()
    
    # Load the mp3 file
    pygame.mixer.music.load("welcome.mp3")
    
    # Play the loaded mp3 file
    pygame.mixer.music.play()
    
    # Wait loop to keep the script running until the audio finishes
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Example usage
speak_text('Welcome to Project Aria')
