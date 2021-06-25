# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:49:20 2020

@author: Onur Tunc
"""
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
from skimage import  data,io, filters, exposure, img_as_float
from skimage import transform, morphology
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from skimage.segmentation import active_contour
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Arayüzü oluşturuyoruz
root = Tk()
root.title("Image Processing P1")
default=Image.open("default.png")
deff=ImageTk.PhotoImage(default)
panel = Label(root, image = deff)
panel.pack()
img=Image.open("default.png")
img = img.resize((500  , 500), Image.ANTIALIAS)


#RGB image'ı siyah-beyaza çevirmeye yarayan metod
def toGray(image):
    if(image.shape[-1]==3):
        gray=image[:,:,0]
    else:
        gray=image
    return gray

#Active Contour örnek
I = data.astronaut()
I = rgb2gray(I)

s = np.linspace(0, 2*np.pi, 400)
r = 100 + 100*np.sin(s)
c = 220 + 100*np.cos(s)
init = np.array([r, c]).T

snake = active_contour(filters.gaussian(I, 3),
                       init, alpha=0.015, beta=10, gamma=0.001,coordinates='rc')

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(I, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, I.shape[1], I.shape[0], 0])
plt.show()
photo = Tk()
photo.title("Active Contour Example")
canvas = FigureCanvasTkAgg(fig, master=photo)
canvas.draw()
canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
canvas._tkcanvas.pack(side="top", fill="both", expand=1)

#Bilgisayardan bir fotoğrafı açma metodu
def open_img():
    global pimg
    global img
    img = Image.open(filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("png files",".png"),("all files",".*"))))
    img = img.resize((500  , 500), Image.ANTIALIAS) 
    pimg = ImageTk.PhotoImage(img)
    panel.configure(image=pimg)
#Fotoğraf kaydetme metodu
def save_img(img):
    filename=   filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files",".jpg"),("all files",".*")))
    img.save(filename)
   
def apply_changes(tk,new_image):
    global img
    img=new_image
    tk.destroy()   
def apply_tk(image):
    apply=Tk()
    apply.title("Apply")
    label=Label(apply,text="Do you want to apply changes")
    button=Button(apply,text="Yes",command=lambda:apply_changes(apply,image))
    button2=Button(apply,text="No",command=lambda:apply.destroy())
    label.pack()
    button.pack()
    button2.pack()
    apply.mainloop()
    
def configure_img(arr):
    a=Image.fromarray((arr*255).astype(np.uint8))
    imgg=ImageTk.PhotoImage(a)
    panel.configure(image=imgg)
    panel.image=imgg
    panel.pack()
    apply_tk(a)
 
    
@adapt_rgb(each_channel)
def gaussian_each(image):
    return filters.gaussian(image,sigma=3.5)  
@adapt_rgb(each_channel)
def hessian_each(image):
    return filters.hessian(image,mode="constant")
@adapt_rgb(each_channel)
def gabor_each(image):
    return filters.gabor(image,frequency=0.2)
@adapt_rgb(each_channel)
def laplace_each(image):
    return filters.laplace(image)  
@adapt_rgb(each_channel)
def meijering_each(image):
    return filters.meijering(image)  
@adapt_rgb(each_channel)
def sato_each(image):
    return filters.sato(image,mode="constant")  
@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel_h(image)  
@adapt_rgb(each_channel)
def scharr_each(image):
    return filters.scharr(image)  
@adapt_rgb(each_channel)
def prewitt_each(image):
    return filters.prewitt_v(image)  
@adapt_rgb(each_channel)
def median_each(image):
    return filters.median(image)
def gaussian_filter(image):
    im_frame=np.array(image)
    gs=gaussian_each(im_frame)
    io.imshow(gs)
    io.show()
    configure_img(gs)
def laplace_filter(image):
    im_frame=np.array(image)
    fr= laplace_each(im_frame)
    configure_img(fr)  
def gabor_filter(image):
    im_frame=np.array(image)
    filt_real,filt_imag=gabor_each(im_frame)
    plt.figure()
    configure_img(filt_real)
def hessian_filter(image):
    im_frame=np.array(image)
    hs=hessian_each(im_frame)
    io.imshow(hs)
    io.show()
    configure_img(hs)
def meijering_filter(image):
    im_frame=np.array(image)
    mj=meijering_each(im_frame)
    io.imshow(mj)
    io.show()
    configure_img(mj)   
def sato_filter(image):
    im_frame=np.array(image)
    st=sato_each(im_frame)
    io.imshow(st)
    io.show()
    configure_img(st)
def sobel_filter(image):
    im_frame=np.array(image)
    sb=sobel_each(im_frame)
    io.imshow(sb)
    io.show()
    configure_img(sb)
def scharr_filter(image):
    im_frame=np.array(image)
    sh=scharr_each(im_frame)
    io.imshow(sh)
    io.show()
    configure_img(sh)
def prewitt_filter(image):
    im_frame=np.array(image)
    pt=prewitt_each(im_frame)
    io.imshow(pt)
    io.show()
    configure_img(pt)
def median_filter(image):
    im_frame=np.array(image)
    rbrt=median_each(im_frame)
    io.imshow(rbrt)
    io.show()
    configure_img(rbrt)
      
#Filtrelemeler bitiş

#Histogram görüntüleme ve eşitleme başlangıç
def histogram_show(img2):
    histtk = Tk()
    histtk.title("Histogram")
    narrayimg = np.array(img2)
    image = img_as_float(narrayimg)
    hist = exposure.histogram(image)
    hist = np.array(hist)
    img3 = Image.fromarray((hist*255)).resize((300,300))
    pimg = ImageTk.PhotoImage(image = img3,master = histtk)
    panel = Label(histtk,image = pimg)
    panel.image = pimg
    panel.pack()
    histtk.mainloop()
def histogram_equalize(image):
    imgg=np.array(image)
    img=img_as_float(imgg)
    img_ex=exposure.equalize_hist(img)
    configure_img(img_ex)  
#Soru 5(Uzaysal dönüşüm işlemleri)   
def img_resize(image):
    imresize=Tk()
    imresize.title("Resize")
    x=Entry(imresize)
    x.pack()
    y=Entry(imresize)
    y.pack()
    button=Button(imresize,text="Resize",command=lambda:resize2(image,int(x.get()),int(y.get())))
    button.pack()
    imresize.mainloop()    
def resize2(image,x,y):
    im_frame=np.array(image)
    new_img=transform.resize(im_frame,(x,y))
    configure_img(new_img)
def img_rotate(image):
    imrotate=Tk()
    imrotate.title("Rotate")
    x=Entry(imrotate)
    x.pack()
    bt=Button(imrotate,text="Rotate",command=lambda:rotate2(image,int(x.get())))
    bt.pack()
    imrotate.mainloop()
    
def rotate2(image,angle):
    im_frame=np.array(image)
    new_img=transform.rotate(im_frame,angle)
    configure_img(new_img)
def img_swirl(image):
    imswirl=Tk()
    imswirl.title("Swirl")
    x=Entry(imswirl)
    x.pack()
    bt=Button(imswirl,text="Swirl",command=lambda:swirl2(image,int(x.get())))
    bt.pack()
    imswirl.mainloop()   
def swirl2(image,strn):
    im_frame=np.array(image)
    new_img=transform.swirl(im_frame,rotation=0,radius=100,strength=strn)
    configure_img(new_img) 
def img_warp(image):
    im_frame=np.array(image)
    matrix = np.array([[1.2, 0, 0], [0.1, 1, -10], [0, 0, 1]])
    warped = transform.warp(im_frame, matrix)
    configure_img(warped)
     
def pyramid(image):
    im_frame=np.array(image)
    new_img=transform.pyramid_expand(toGray(im_frame),upscale=1.5)
    configure_img(new_img)

#Soru 6    
#Yoğunluk Dönüşümü işlemleri
def intensity_in(image):
    imintensity=Tk()
    imintensity.title("Intensity")
    x=Entry(imintensity)
    x.pack()
    bt=Button(imintensity,text="Change Intensity",command=lambda:intensity_in2(image,int(x.get())))
    bt.pack()
    imintensity.mainloop()      
def intensity_in2(image,x):
    im_frame=np.array(image)
    img_new=exposure.rescale_intensity(im_frame,in_range=(0,x))
    configure_img(img_new)
def intensity_out(image):
    imintensity=Tk()
    imintensity.title("Intensity")
    x=Entry(imintensity)
    x.pack()
    bt=Button(imintensity,text="Change Intensity",command=lambda:intensity_out2(image,int(x.get())))
    bt.pack()
    imintensity.mainloop()      
def intensity_out2(image,x):
    im_frame=np.array(image)
    img_new=exposure.rescale_intensity(im_frame,out_range=(x,0))
    configure_img(img_new)
#Morfolojik işlemler
def areaClosing(image):
    im_frame=np.array(image)
    new_img=morphology.area_closing(toGray(im_frame))
    configure_img(new_img)
def areaOpening(image):
    im_frame=np.array(image)
    new_img=morphology.area_opening(toGray(im_frame))
    configure_img(new_img)  
def binaryClosing(image):
    im_frame=np.array(image)
    new_img=morphology.binary_closing(im_frame)
    configure_img(new_img)
def binaryOpening(image):
    im_frame=np.array(image)
    new_img=morphology.binary_opening(im_frame)
    configure_img(new_img)
def binaryDilation(image):
    im_frame=np.array(image)
    new_img=morphology.binary_dilation(im_frame)
    configure_img(new_img)
def binaryErosion(image):
    im_frame=np.array(image)
    new_img=morphology.binary_erosion(im_frame)
    configure_img(new_img)
def blackTophat(image):
    im_frame=np.array(image)
    new_img=morphology.black_tophat(im_frame)
    configure_img(new_img)
def diameterClosing(image):
    im_frame=np.array(image)
    new_img=morphology.diameter_closing(toGray(im_frame))
    configure_img(new_img)
def diameterOpening(image):
    im_frame=np.array(image)
    new_img=morphology.diameter_opening(toGray(im_frame))
    configure_img(new_img)
def localMaxima(image):
    im_frame=np.array(image)
    new_img=morphology.local_maxima(im_frame)
    configure_img(new_img)
def myFilter(image):
    im_frame=np.array(image)
    img=img_as_float(im_frame)
    img_ex=exposure.equalize_hist(img)
    img_ex=transform.pyramid_expand(toGray(img_ex),upscale=1.2)
    img_ex=Image.fromarray((img_ex*255).astype(np.uint8))
    img_ex=np.array(img_ex)
    new_img=median_each(img_ex)
    configure_img(new_img)
    
    

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open Image", command = open_img)
filemenu.add_command(label="Save Image",command=lambda:save_img(img))
filemenu.add_separator()
filemenu.add_command(label="Exit")
menubar.add_cascade(label="File", menu=filemenu)

filtermenu= Menu(menubar, tearoff=0)
filtermenu.add_command(label="Gaussian filter",command = lambda:gaussian_filter(img))
filtermenu.add_command(label="Gabor filter",command = lambda:gabor_filter(img))
filtermenu.add_command(label="Hessian filter",command = lambda:hessian_filter(img))
filtermenu.add_command(label="Meijering filter",command = lambda:meijering_filter(img))
filtermenu.add_command(label="Sato filter",command = lambda:sato_filter(img))
filtermenu.add_command(label="Sobel filter",command = lambda:sobel_filter(img))
filtermenu.add_command(label="Scharr filter",command = lambda:scharr_filter(img))
filtermenu.add_command(label="Prewitt filter",command = lambda:prewitt_filter(img))
filtermenu.add_command(label="Median filter",command = lambda:median_filter(img))
filtermenu.add_command(label="Laplace filter",command = lambda:laplace_filter(img))
menubar.add_cascade(label="Filters", menu=filtermenu)

histogrammenu=Menu(menubar,tearoff=0)
histogrammenu.add_command(label="Histogram Eşitleme",command = lambda:histogram_equalize(img))
histogrammenu.add_command(label="Histogram Görüntüleme",command=lambda:histogram_show(img))
menubar.add_cascade(label="Histogram", menu=histogrammenu)

transformmenu=Menu(menubar,tearoff=0)
transformmenu.add_command(label="Resize",command = lambda:img_resize(img))
transformmenu.add_command(label="Rotate",command = lambda:img_rotate(img))
transformmenu.add_command(label="Swirl",command = lambda:img_swirl(img))
transformmenu.add_command(label="Warp",command = lambda:img_warp(img))
transformmenu.add_command(label="Pyramid expand",command = lambda:pyramid(img))
menubar.add_cascade(label="Transform",menu=transformmenu)

intensitymenu=Menu(menubar,tearoff=0)
intensitymenu.add_command(label="Intensity in",command = lambda:intensity_in(img))
intensitymenu.add_command(label="Intensity out",command = lambda:intensity_out(img))
menubar.add_cascade(label="Intensity",menu=intensitymenu)


morfmenu=Menu(menubar,tearoff=0)
morfmenu.add_command(label="Area closing",command = lambda:areaClosing(img))
morfmenu.add_command(label="Area opening",command = lambda:areaOpening(img))
morfmenu.add_command(label="Binary closing",command = lambda:binaryClosing(img))
morfmenu.add_command(label="Binary dilation",command = lambda:binaryDilation(img))
morfmenu.add_command(label="Binary erosion",command = lambda:binaryErosion(img))
morfmenu.add_command(label="Binary opening",command = lambda:binaryOpening(img))
morfmenu.add_command(label="Black tophat",command = lambda:blackTophat(img))
morfmenu.add_command(label="Diameter opening",command = lambda:diameterOpening(img))
morfmenu.add_command(label="Diameter closing",command = lambda:diameterClosing(img))
morfmenu.add_command(label="Local Maxima",command = lambda:localMaxima(img))

menubar.add_cascade(label="Morfolojik işlemler",menu=morfmenu)

mymenu=Menu(menubar,tearoff=0)
mymenu.add_command(label="My Filter:Ghosting",command=lambda:myFilter(img))
menubar.add_cascade(label="My Filter",menu=mymenu)

root.config(menu=menubar)
root.mainloop()


    

    
    

    


