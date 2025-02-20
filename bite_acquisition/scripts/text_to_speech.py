import pygame
from gtts import gTTS
import tempfile

def say(text):
    # Create a temporary file to store the audio
    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as temp_audio_file:
        # Convert the text to speech
        tts = gTTS(text=text, lang='en')
        
        # Save the audio to the temporary file
        tts.save(temp_audio_file.name)
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load and play the audio file
        pygame.mixer.music.load(temp_audio_file.name)
        pygame.mixer.music.play()

        # Wait for the audio to finish before returning
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

if __name__ == "__main__":
    # Example usage
    say("Hello, this is a test.")
    print("LMAOLMAOLMAO")
    say("This is the second say")