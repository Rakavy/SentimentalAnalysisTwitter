from tkinter import *


class Example(Frame):
  
    def __init__(self, parent):
        Frame.__init__(self, parent, background="white")   
         
        self.parent = parent
        self.initUI()
        
    
    def initUI(self):
      
        self.parent.title("Twitter Sentimental Analysis")
        self.pack(fill=BOTH, expand=True)
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)
        
        lbl = Label(self, text="Search Term")
        lbl.grid(sticky=E, pady=4, padx=5)
        
        area = Text(self)
        area.grid(row=1, column=0, columnspan=2, rowspan=4, 
            padx=5, sticky=E+W+S+N)
        
        abtn = Button(self, text="Activate")
        abtn.grid(row=1, column=3)

        cbtn = Button(self, text="Close")
        cbtn.grid(row=2, column=3, pady=4)
        
        hbtn = Button(self, text="Help")
        hbtn.grid(row=5, column=0, padx=5)

        obtn = Button(self, text="OK")
        obtn.grid(row=5, column=3)         

def on_click(event):
	

def main():
  
    root = Tk()
    root.geometry("1000x800+600+250")
    app = Example(root)
    root.mainloop()  


if __name__ == '__main__':
    main()