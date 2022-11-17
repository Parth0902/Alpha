import pyttsx3


# define talk function
def talk(text):
    engine=pyttsx3.init("sapi5")
    voices=engine.getProperty('voices')
    engine.setProperty('voices',voices[0].id)
    engine.setProperty('rate',170)
    print("  ")
    print(f"Ass :{text}")
    engine.say(text=text)
    engine.runAndWait()
    print("  ")






