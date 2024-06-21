import pvporcupine
import pyaudio
import speech_recognition as sr
import numpy as np
import cv2

ACCESS_KEY = '2HEVbzGtkFD6kt8RmmIoiIrZLdzT6ViysO/an4cKZoo9Taw2vx/76A=='

def recognize_speech(timeout=7):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for commands...")
        try:
            audio = recognizer.listen(source, timeout=timeout)
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for command")
            return None

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Recognized command: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def main():
    porcupine = pvporcupine.create(access_key=ACCESS_KEY, keywords=["computer"])
    pa = pyaudio.PyAudio()

    default_device_index = pa.get_default_input_device_info()['index']

    def create_audio_stream():
        return pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
            input_device_index=default_device_index 
        )

    try:
        print("Listening for wake word...")
        while True:
            stream = create_audio_stream()
            while True:
                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = np.frombuffer(pcm, dtype=np.int16)

                if porcupine.process(pcm) >= 0:
                    print("Wake word detected")
                    stream.stop_stream()
                    stream.close()

                    command = recognize_speech()
                    if command:
                        if "capture image" in command:
                            print("Capture Image")
                        elif "convert image to text" in command:
                            print("Convert Image to Text")
                        elif "text to speech" in command:
                            print("Text to Speech")
                        elif "stop" in command:
                            print("Stop command detected, exiting program...")
                            return                    
                    break  

            print("Listening for wake word...")
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                print("Exiting program...")
                break

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()