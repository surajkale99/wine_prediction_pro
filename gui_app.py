import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tkinter import *
import pickle
from PIL import Image, ImageTk

win=Tk()
win.title('Wine Quality Prediction System')
win.geometry('600x600')
win.config(bg='gray')


# Load the background image
bg_image = Image.open("wineimg.png")  # Replace with your actual image file
bg_image = bg_image.resize((1588, 1600), Image.LANCZOS)  # Resize to match the window size
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Canvas to place the background image
canvas = Canvas(win, width=1500, height=1600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")  # Set image as background
def disp():
    d1 = float(var1.get())
    d2 = float(var2.get())
    d3 = float(var3.get())
    d4 = float(var4.get())
    d5 = float(var5.get())
    d6 = float(var6.get())
    d7 = float(var7.get())
    d8 = float(var8.get())
    d9 = float(var9.get())
    d10 = float(var10.get())
    d11 = float(var11.get())

    list1=[[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11]]

    with open('wine_sc','rb') as f:
        sc=pickle.load(f)
    with open('wine_model','rb') as f:
        model=pickle.load(f)
    a_sc=sc.transform(list1)
    result=model.predict(a_sc)
    if result[0]==1:
        z='Its a Good Quality Wine'
    else:
        z='Its a Average Quality Wine'
    l12.config(text=z)



# main label
l0=Label(win,text='WINE QUALITY PREDICTION SYSTEM',bg='white',fg='black',width=40,bd=5,relief='ridge',
         font=('times new roman',18,'bold'))
l0.place(x=200,y=10)


# fixed acidity

l1=Label(win,text='Fixed Acidity',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l1.place(x=100,y=60)

var1=StringVar()
e1=Entry(win,textvariable=var1,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e1.place(x=340,y=60)

# volatile acidity

l2=Label(win,text='Volatile Acidity',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l2.place(x=100,y=100)

var2=StringVar()
e2=Entry(win,textvariable=var2,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e2.place(x=340,y=100)

# citric acid

l3=Label(win,text='Citric Acid',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l3.place(x=100,y=140)

var3=StringVar()
e3=Entry(win,textvariable=var3,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e3.place(x=340,y=140)

# residual sugar

l4=Label(win,text='Residual Sugar',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l4.place(x=100,y=180)

var4=StringVar()
e4=Entry(win,textvariable=var4,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e4.place(x=340,y=180)

# chlorides

l5=Label(win,text='Chlorides',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l5.place(x=100,y=220)

var5=StringVar()
e5=Entry(win,textvariable=var5,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e5.place(x=340,y=220)

# free sulfur dioxide

l6=Label(win,text='Free Sulfur Dioxide',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l6.place(x=100,y=260)

var6=StringVar()
e6=Entry(win,textvariable=var6,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e6.place(x=340,y=260)

# total sulfur dioxide

l7=Label(win,text='Total Sulfur Dioxide',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l7.place(x=100,y=300)

var7=StringVar()
e7=Entry(win,textvariable=var7,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e7.place(x=340,y=300)

# density

l8=Label(win,text='Density',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l8.place(x=100,y=340)

var8=StringVar()
e8=Entry(win,textvariable=var8,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e8.place(x=340,y=340)

# pH

l9=Label(win,text='pH',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l9.place(x=100,y=380)

var9=StringVar()
e9=Entry(win,textvariable=var9,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e9.place(x=340,y=380)


# sulphates
l10=Label(win,text='Sulphates',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l10.place(x=100,y=420)

var10=StringVar()
e10=Entry(win,textvariable=var10,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e10.place(x=340,y=420)

#alcohol

l11=Label(win,text='Alcohol',bg='white',fg='black',width=20,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))
l11.place(x=100,y=460)

var11=StringVar()
e11=Entry(win,textvariable=var11,bg='white',fg='black',width=30,bd=5,relief='ridge',font=('times new roman',14,'bold'))
e11.place(x=340,y=460)


# button

b1=Button(win,command=disp,text='Predict Quality',bg='black',fg='white',width=12,bd=5,relief='ridge',
         font=('times new roman',14,'bold'))

b1.place(x=340,y=520)


# extra label

l12=Label(win,text='Quality of Wine is',bg='green',fg='white',height=6,width=25,bd=5,relief='ridge',
         font=('times new roman',18,'bold'))
l12.place(x=800,y=260)



win.mainloop()