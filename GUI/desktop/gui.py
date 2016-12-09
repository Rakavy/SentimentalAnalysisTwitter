from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from tkinter import *
from twitter import twitterApi
import matplotlib
import sys

matplotlib.use("TkAgg")



f = Figure(figsize=(6, 4), dpi=100)
a = f.add_subplot(111)


def analyze(x):
    if (len(x) > 100):
        if (len(x) % 2 == 1):
            return 0
        elif (len(x) % 2 == 0):
            return 4
    else:
        return 2


def searchTweet(*args):
    global f
    global a
    colors = ['#00FF40', '#00BFFF', '#FF0000']
    pos = 0
    neg = 0
    neu = 0
    word = tweet.get()
    word.replace(" ", "+")
    word.replace("#", "%23")

    listTweets = twitterApi(word)
    lblTweets.delete('1.0', END)

    for x in listTweets:
        x = ''.join(c for c in x if c <= '\uFFFF')
        z = analyze(x)
        if (z == 0):
            lblTweets.insert(END, x + '\n', 'neg')
            neg += 1
        elif (z == 4):
            lblTweets.insert(END, x + '\n', 'pos')
            pos += 1
        else:
            lblTweets.insert(END, x + '\n', 'neu')
            neu += 1

    sizes = [pos, neu, neg]
    labels = 'Positive: '+str(pos), 'Neutral: '+str(neu), 'Negative: '+str(neg)
    a.clear()
    a.pie(sizes, labels=labels, colors=colors,
          autopct='%1.1f%%', shadow=True, startangle=90)
    a.axis('equal')
    figure.draw()

    searchWord.set(word)


def closeGui():
    sys.exit()

root = Tk()
root.geometry("1300x700+400+20")
mainFrame = Frame(root, background="white")

root.title("Twitter Sentimental Analysis")
mainFrame.pack(fill=BOTH, expand=False)

mainFrame.columnconfigure(0, weight=1)
mainFrame.columnconfigure(3, pad=7)
mainFrame.rowconfigure(5, weight=1)

tweet = StringVar()
searchWord = StringVar()

lblSearch = Label(mainFrame, text="Word:", width=10)
lblSearch.grid(row=2, column=0, sticky=E, pady=10)

abtn = Button(mainFrame, text="Search", width=8, command=searchTweet)
abtn.grid(row=2, column=1, pady=20, sticky=E)

search = Entry(mainFrame, width=45, textvariable=tweet)
search.grid(row=2, column=1, sticky=W, pady=20, padx=10)

searchTweets = Label(mainFrame, text="Searched Word:")
searchTweets.grid(row=3, column=0, sticky=E, padx=10)

searchTweets = Label(mainFrame, textvariable=searchWord, width=10)
searchTweets.grid(row=3, column=1, padx=10)

lstTweets = Label(mainFrame, text="List of tweets:", width=12)
lstTweets.grid(row=4, column=0, sticky=S, padx=10)

frame3 = Frame(mainFrame)
frame3.grid(row=5, column=0, sticky='W', padx=10)

lblTweets = Text(frame3, width=60)
lblTweets.grid(row=5, column=0, padx=10, sticky='W')
lblTweets.tag_config("pos", background="white", foreground="#458B00")
lblTweets.tag_config("neg", background="white", foreground="#FF0000")
lblTweets.tag_config("neu", background="white", foreground="#0000CD")

scroll = Scrollbar(frame3, orient=VERTICAL)
lblTweets.config(yscrollcommand=scroll.set)
scroll.config(command=lblTweets.yview)
mainFrame.grid()
scroll.grid(column=0, row=5, sticky=N+S+E)
#  lblTweets.grid(side=LEFT,  fill=BOTH, expand=1)

figure = FigureCanvasTkAgg(f, master=mainFrame)
figure.show()
figure.get_tk_widget().grid(row=5, column=1, padx=10, sticky='E')

cbtn = Button(mainFrame, text="Close", width=10, command=closeGui)
cbtn.grid(row=6, column=1,  pady=4)

search.focus()
root.bind('<Return>', searchTweet)

root.mainloop()
