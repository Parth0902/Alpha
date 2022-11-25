import datetime
import ecapture
import wolframalpha
from talk import talk
from listen import listen
def Time():
    time=datetime.datetime.now().strftime("%H:%M")
    talk(time)

def Day():
    day=datetime.datetime.now().strftime("%A")
    talk(day)

def Date():
    date=datetime.date.today()
    talk(date)

def Note():
  
        talk("What should i write, sir")
        note = listen()
        file = open('alpha.txt', 'a')
        talk("Sir, Should i include date and time")
        snfm = listen()
        if 'yes' in snfm or 'sure' in snfm:
            strTime = datetime.datetime.now().strftime("%H:%M")
            strDay=datetime.datetime.now().strftime("%A")
            file.write(strDay,)
            file.write(strTime)
            file.write("\n")
            file.write(" :- ")
            file.write(note)
            file.write("\n")
        else:
            file.write(note)
        
        
def showNote():        
        talk("Showing Notes")
        file = open("alpha.txt", "r")
        print(file.read())
        talk(file.read(6))
    

def nonINputExecution(query):
    
    query=str(query)

    if "time" in query:
        Time()

    elif "date" in query:
        Date()
    
    elif "day" in query:
        Day()
  
    elif "note" in query:
        Note()

    elif "showNote" in query:
        showNote()

  

def InputExecution(tag,query):
     
    if "wikipedia" in tag:
     name=str(query).replace("who is","").replace("tell me about","").replace("what is","").replace("wikipedia","")
     import wikipedia
     result = wikipedia.summary(name)
     talk(result)

    elif "google" in tag:
        query=str(query).replace("google","")
        query=str(query).replace("search","")
        import pywhatkit
        pywhatkit.search(query)

    