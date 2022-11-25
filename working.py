import datetime
import ecapture
import wolframalpha
from talk import talk
def Time():
    time=datetime.datetime.now().strftime("%H:%M")
    talk(time)

def Day():
    day=datetime.datetime.now().strftime("%A")
    talk(day)

def Date():
    date=datetime.date.today()
    talk(date)

def nonINputExecution(query):
    
    query=str(query)

    if "time" in query:
        Time()

    elif "date" in query:
        Date()
    
    elif "day" in query:
        Day()



  

def InputExecution(tag,query):
     
    if "wikipedia" in tag:
     name=str(query).replace("","")
     import wikipedia
     result = wikipedia.summary(name)
     talk(result)

    elif "what is" in tag or "who is" in tag:
      client = wolframalpha.Client("API_ID")
      res = client.query(query)
             
      try:
       print (next(res.results).text)
       talk(next(res.results).text)
      except StopIteration:
       print ("No results")