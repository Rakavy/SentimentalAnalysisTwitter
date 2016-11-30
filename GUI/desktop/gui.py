from tkinter import *
from twitter import twitterApi 

def searchTweet(*args):
    listTweets = twitterApi(tweet.get())
    
    for x in listTweets:
        lblTweets.insert(END, x + '\n')

    lblTweets.pack()
    searchWord.set(tweet.get())
    #foundTweet.set(listTweets)
    
#def on_click(event):

root = Tk()
root.geometry("1000x800+600+250")
mainFrame = Frame(root,background="white")
    
root.title("Twitter Sentimental Analysis")
mainFrame.pack(fill=BOTH, expand=True)

mainFrame.columnconfigure(0, weight=1)
mainFrame.columnconfigure(3, pad=7)

tweet = StringVar() 
searchWord = StringVar()

lblSearch = Label(mainFrame, text="Word:", width=10)
lblSearch.grid(row=2,column =0, pady=20)

search = Entry(mainFrame,text="get", width=80, textvariable = tweet)
search.grid(row=2, column=1, pady=20, padx=10)
    
abtn = Button(mainFrame, text="Go",width=4, command = searchTweet)
abtn.grid(row=2, sticky=W,column=2,pady=20)

searchTweets = Label(mainFrame, text="Searched Word:")
searchTweets.grid(row=3,column =0,padx=10)

searchTweets = Label(mainFrame, textvariable = searchWord, width=10)
searchTweets.grid(row=3,column =1, padx=10)

lstTweets = Label(mainFrame, text="List of tweets:", width=10)
lstTweets.grid(row=4,column =0,padx=10)

lblTweets = Text(mainFrame,  width=150)
lblTweets.grid(row=5,column =0, padx=10)

cbtn = Button(mainFrame, text="Close")
cbtn.grid(row=6, column=2, pady=4)

lblEmpty = Label(mainFrame, text="", width=5)
lblEmpty.grid(row=2,column =3, pady=20)

search.focus()
root.bind('<Return>', searchTweet)

root.mainloop()
