import pyttsx3
import speech_recognition as sr
from model import Model

# Method that says a line of text using pyttsx3
def say_something(text: str) -> None:
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
# Creates an infinite loop where chatbot is listening for voice input
def main():
    r = sr.Recognizer()
    model = Model()
    while(1):
        try:
            with sr.Microphone(device_index=4) as src2:
                r.adjust_for_ambient_noise(src2, duration=0.2)
                audio2 = r.listen(src2)
                text = r.recognize_google(src2).lower()

                print("Picked up: ", text)
                say_something(model.answer(text))
        except sr.RequestError as e:
            print("Could not get results!")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
