from tkinter import *
from twitter import twitterApi 


def searchTweet(*args):
    word = tweet.get()
    word.replace(" ","+")
    word.replace("#", "%23")
    listTweets = twitterApi(word)
    lblTweets.delete('1.0', END) 

    for x in listTweets:
        x = ''.join(c for c in x if c <= '\uFFFF')
        lblTweets.insert(END, x + '\n')

    searchWord.set(word)
    #foundTweet.set(listTweets)
    
#def on_click(event):

def convert65536(s):
    #Converts a string with out-of-range characters in it into a string with codes in it.
    l=list(s);
    i=0;
    while i<len(l):
        o=ord(l[i]);
        if o>65535:
            l[i]="{"+str(o)+"ū}";
        i+=1;
    return "".join(l);

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

lblTweets = Text(mainFrame, width=150)
lblTweets.grid(row=5,column =0, padx=10)

cbtn = Button(mainFrame, text="Close")
cbtn.grid(row=6, column=2, pady=4)

lblEmpty = Label(mainFrame, text="", width=5)
lblEmpty.grid(row=2,column =3, pady=20)

search.focus()
root.bind('<Return>', searchTweet)

root.mainloop()
