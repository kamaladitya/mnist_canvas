import tkinter as tk
import io
from PIL import Image
import numpy as np
import cv2
#from test import test


def save():
	ps = canvas.postscript(colormode='color')
	img = Image.open(io.BytesIO(ps.encode('utf-8')))
	img.save('test.jpg')
	full_image=cv2.imread("test.jpg")
	img=cv2.resize(full_image,(28,28))
	img=Image.fromarray(img)
	img.save("test.jpg")
	img =Image.open("test.jpg").convert("L")
	arr=np.array(img)
	#print(arr)
	#test(img)

def paint(event):
        python_black = "#000000"
        x1, y1 = ( event.x - 1 ), ( event.y - 1 )
        x2, y2 = ( event.x + 1 ), ( event.y + 1 )
        canvas.create_oval( x1, y1, x2, y2, fill = python_black, width=10)


line_start = None
canvas = tk.Canvas(width=200, height=200, bg="white")
canvas.bind("<B1-Motion>", paint)
button = tk.Button( text="save",command=save)
canvas.pack()
button.pack(pady=1)

tk.mainloop()

