from tkinter import *
from twitter import twitterApi
import sys
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure



#pi chart labels
labels = 'Positive', 'Neutral', 'Negative'
sizes = [15, 30, 20]
colors = ['#00FF40','#00BFFF', '#FF0000']

f = Figure(figsize=(5,4),dpi=100)
a=f.add_subplot(111)

def searchTweet(*args):
    word = tweet.get()
    word.replace(" ","+")
    word.replace("#", "%23")
    global labels
    global colors
    global f
    global a
    sizes = [20,10,10]
    a.clear()
    a.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    a.axis('equal')
    figure.draw()
    listTweets = twitterApi("word")
    lblTweets.delete('1.0', END) 

    for x in listTweets:
        x = ''.join(c for c in x if c <= '\uFFFF')     
        lblTweets.insert(END, x + '\n')
        lblTweets.configure(bg=colors[0])

    searchWord.set(word)

def closeGui():
    sys.exit()

root = Tk()
root.geometry("1000x800+800+20")
mainFrame = Frame(root,background="white")
    
root.title("Twitter Sentimental Analysis")
mainFrame.pack(fill=BOTH, expand=True)

mainFrame.columnconfigure(0, weight=1)
mainFrame.columnconfigure(3, pad=7)
mainFrame.rowconfigure(5, weight=1)

tweet = StringVar() 
searchWord = StringVar()

lblSearch = Label(mainFrame, text="Word:", width=10)
lblSearch.grid(row=2,column =0, sticky=E, pady=10)
    
abtn = Button(mainFrame, text="Search",width=6, command = searchTweet)
abtn.grid(row=2, column=1,pady=20,sticky=E)

search = Entry(mainFrame,width=45, textvariable = tweet)
search.grid(row=2, column=1, sticky=W, pady=20, padx=10)

searchTweets = Label(mainFrame, text="Searched Word:")
searchTweets.grid(row=3,column =0, sticky=E, padx=10)

searchTweets = Label(mainFrame, textvariable = searchWord, width=10)
searchTweets.grid(row=3,column =1, padx=10)

lstTweets = Label(mainFrame, text="List of tweets:", width=10)
lstTweets.grid(row=4,column =0, sticky=S,padx=10)

frame3 = Frame(mainFrame)  
frame3.grid(row=5,column =0, sticky=N, padx=10)
scroll = Scrollbar(frame3, orient=VERTICAL)

lblTweets = Text(frame3, width=150,yscrollcommand=scroll.set, height=60)
lblTweets.grid(row=5,column =0, sticky=N, padx=5)

scroll.config (command=lblTweets.yview)
scroll.pack(side=RIGHT, fill=Y)
lblTweets.pack(side=LEFT,  fill=BOTH, expand=1)

figure = FigureCanvasTkAgg(f, master=mainFrame)
figure.show()
figure.get_tk_widget().grid(row=5, column =1, padx=10)

cbtn = Button(mainFrame, text="Close", width=10,command=closeGui)
cbtn.grid(row=6, column=1,  pady=4)

search.focus()
root.bind('<Return>', searchTweet)

root.mainloop()