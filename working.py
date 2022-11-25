import datetime
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
