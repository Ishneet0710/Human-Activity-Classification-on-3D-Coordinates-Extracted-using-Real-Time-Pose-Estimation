from tkinter import *
from PIL import Image, ImageTk
import cv2


win = Tk()
win.geometry("640x480")
label = Label(win)
label.grid(row=0, column=0)
cap = cv2.VideoCapture(0)


def camera():
   cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   label.after(20, camera)


win.title("Output of Webcam")
camera()
win.mainloop()