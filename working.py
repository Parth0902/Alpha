import datetime
from talk import talk
def Time():
    time=datetime.now().strftime("%H:%M")
    talk(time)

def date():
    date=datetime.date.today()
    talk(date)

def nonINputExecution(query):
    query=str(query)

    if "time" in query:
        Time()

    elif "date" in query:
        date()
    