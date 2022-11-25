
import speech_recognition as sr


def listen():

    # here r represents the collection of speech recognition functionalities

    r=sr.Recognizer()
    
    # we are inputing voice from microphone and storing it as audio
     
    with sr.Microphone() as source:
        print("Listening..............")
        r.pause_threshold=1
        audio=r.listen(source,0,3)

    try:

    #   Here we are recognizing the audio with help of google recognizer and printing it as query
      print("recognizing.............")
      query= r.recognize_google(audio,language="en-in")
      print(f"you said: {query}")    
    
    except:
        return ""
    
    # typecasting query as string
    query=str(query)


    # returning the query by converting it into lower case
    return query.lower()

