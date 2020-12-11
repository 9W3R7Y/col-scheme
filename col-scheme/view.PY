import colorsys
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader,Dataset
from PIL import Image, ImageDraw, ImageFont
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as plc
import os
import shutil
from tqdm import tqdm
import glob
import itertools 
import colormath
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import HSVColor, LabColor
from colormath.color_conversions import convert_color
from matplotlib import patheffects as path_effects
from matplotlib.font_manager import FontProperties

def undivide(color_pallet,level = 24):
    divided = list(color_pallet)
    undivided = np.zeros((4,3))#配列の準備
    for n in range(len(color_pallet)):
        undivided[n,0] = np.argmax(color_pallet[n][:level])/24
        undivided[n,1]=color_pallet[n,24]
        undivided[n,2]=color_pallet[n,25]
    return undivided

def rgb(color_pallet):
    for color,n in zip(color_pallet,range(4)):
        [r,g,b] = colorsys.hsv_to_rgb(color[0],color[1],color[2])
        color_pallet[n] = [r,g,b]
    return color_pallet

def hsv(color_pallet):
    for color,n in zip(color_pallet,range(4)):
        [h,s,v] = colorsys.rgb_to_hsv(color[0],color[1],color[2])
        #print(h,s,v)
        color_pallet[n] = [h*360,s*100,v*100]
    return color_pallet

def convert(data,type,target="RGB"):
    if np.max(data)>1:
        for jjj in range(100):
            print("wrong")
    pallets=[]

    for p in data:
        if type == 1 or type== "HSV":
            p = rgb(p)
        elif type == 2 or type== "SHSV":
            p = rgb(undivide(p))
        elif type == 0 or type== "RGB" or type == "RGB_CGAN":
            None
        else:
            print("OMGGGGGGGGGGGGGGG")
            
        if target=="HSV":
            pallets.append(hsv(p))
        else:
            pallets.append(p*255)

    return pallets

def visualize(pallets,iter=-1,show=False,save=False,path="",caption=True):
    x=4
    y=4
    s = 100
    d = 50

    W = x*2*s+(x+1)*d
    H = y*2*s+(y+1)*d

    im = Image.new('RGB', (W,H), (128, 128, 128))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0,0,W,H), fill=(255,255,255))

    def put(pallet,x,y):
        n = 0
        for i in range(2):
            for j in range(2):
                draw.rectangle(((x+i*s,y+j*s),(x+i*s+s,y+j*s+s)),fill=(int(pallet[n,0]),int(pallet[n,1]),int(pallet[n,2])))
                n+=1

    def text(msg,x,y):
        font = ImageFont.truetype("YuGothL.ttc", size=25)
        w, h = draw.textsize(msg,font=font)
        draw.text((x-w/2,y-h/2), msg, fill="black",font=font)
    
    n = 0
    for i in range(x):
        for j in range(y):
            put(pallets[n],i*(2*s+d)+d,j*(2*s+d)+d)
            n+=1

    types=["RGB","HSV","SHSV","RGB_CGAN"]

    if caption==True:
        text(  "Type = "+types[int(type)]+\
                ", lr_G = 10^("+str(np.log10(lr_G))+\
                "), lr_D = 10^("+str(np.log10(lr_D))+\
                "), iter = "+str(iter),W/2,H-d/2)

    if show:
        im.show()

    if save:
        im.save(path)

    return im

def save_all(path,step=1):
    entire = np.load(path+"\\pallet.npy")

    pathimg = path+"\\img\\"
    try:
        shutil.rmtree(pathimg)
    except:
        pass
    os.makedirs(pathimg,exist_ok=True)

    for data,n in zip(entire,tqdm(range(len(entire)))):
        if n%step==0:
            visualize(convert(data,type),str(n*int(iterstep)),save=True,path=pathimg+str(n//step).zfill(4)+".png")

def grid_img(n_current,noNum=True,fileindex=None):
    os.chdir(metapath)
    type = list(map(str,metapath.split("\\")))[-1]

    pathes = glob.glob(metapath+"/G_**_D_**/")
    ordersG = set()
    ordersD = set()

    for text in pathes:
        _,g,d = map(str,text.replace("G_",",").replace("_D_",",").replace("\\","").split(","))
        ordersG.add(g)
        ordersD.add(d)

    ordersG = list(ordersG)
    ordersD = list(ordersD)

    ordersG.sort(reverse=True)
    ordersD.sort()

    numG = len(ordersG)
    numD = len(ordersD)

    fig, axs = plt.subplots(numD,numG,figsize=(7,7),sharex = True, sharey = True)
        
    for order_d,x in zip(ordersD,range(numD)):
        for order_g,y in zip(ordersG,range(numG)):
                path = metapath+"/G_"+order_g+"_D_"+order_d
                [_,_,_,iterstep] = list(np.load(path+"\\metadata.npy")[:,-1])
                data=np.load(path+"/pallet.npy")[n_current]
                axs[x,y].imshow(visualize(convert(data,type),caption=False))
                axs[x,y].axis('off')

    plt.yticks(rotation='vertical')
        
    l=0.08
    r=1-l
    b=0.08
    t=1-b

    dx = (1-2*l)/numG
    dy = (1-2*b)/numD

    for lr,n in zip(reversed(ordersD),range(numD)):
        fig.text(r+0.01,b+dy*(1+2*n)/2, 'η_D : 10^('+str(lr)+")", va='center',rotation='vertical')

    for lr,n in zip(ordersG,range(numG)):
        fig.text(l+dx*(1+2*n)/2,t+0.01, 'η_G : 10^('+str(lr)+")", ha='center')

    plt.subplots_adjust(left=l,right=r,top=t,bottom=b,wspace=0,hspace=0)
    
    if noNum==True:
        plt.savefig(metapath+"\\"+type+"_pallets.png")
        if show == True:plt.show() 
    else:
        fig.text(0.5, 0.03, "Type = "+type+", iter = "+str(n_current*int(iterstep)), ha='center')
        os.makedirs(metapath+"\\img",exist_ok=True)
        plt.savefig(metapath+"\\img\\"+type+"_"+str(fileindex).zfill(4)+".png")
        plt.close()

def grid_err():
    os.chdir(metapath)
    type = list(map(str,metapath.split("\\")))[-1]
    pathes = glob.glob(metapath+"/G_**_D_**/")
    ordersG = set()
    ordersD = set()

    for text in pathes:
        _,g,d = map(str,text.replace("G_",",").replace("_D_",",").replace("\\","").split(","))
        ordersG.add(g)
        ordersD.add(d)

    ordersG = list(ordersG)
    ordersD = list(ordersD)

    ordersG.sort(reverse=True)
    ordersD.sort()

    numG = len(ordersG)
    numD = len(ordersD)

    fig, axs = plt.subplots(numD,numG,figsize=(7,7),sharex = True, sharey = True)
        
    for order_d,x in zip(ordersD,range(numD)):
        for order_g,y in zip(ordersG,range(numG)):
            path = metapath+"/G_"+order_g+"_D_"+order_d
            kernel = np.ones(1000)/1000
            axs[x,y].plot(np.convolve(np.load(path+"/errG.npy"),kernel,mode="same"),label="G",              linestyle="solid",  linewidth=1)
            axs[x,y].plot(np.convolve(np.load(path+"/errDreal.npy"),kernel,mode="same"),label="D_real",   linestyle='dashdot',linewidth=1)
            axs[x,y].plot(np.convolve(np.load(path+"/errDfake.npy"),kernel,mode="same"),label="D_fake",   linestyle='dashed', linewidth=1)
            if x == 0 and y == 0:axs[x,y].legend(loc='upper left')
            axs[x,y].set_yticks([0.0, 0.2,0.4,0.6,0.8])
            axs[x,y].tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)

    plt.yticks(rotation='vertical')
    
    plt.ylim(0,1)

    l=0.08
    r=1-l
    b=0.08
    t=1-b

    dx = (1-2*l)/numG
    dy = (1-2*b)/numD

    fig.text(0.5, 0.02, 'Iteration', ha='center')
    fig.text(0.005, 0.5, 'Error', va='center', rotation='vertical')

    for lr,n in zip(reversed(ordersD),range(numD)):
        fig.text(r+0.01,b+dy*(1+2*n)/2, 'Lr_D : 10^('+str(lr)+")", va='center',rotation='vertical')

    for lr,n in zip(ordersG,range(numG)):
        fig.text(l+dx*(1+2*n)/2,t+0.01, 'η_G : 10^('+str(lr)+")", ha='center')

    plt.subplots_adjust(left=l,right=r,top=t,bottom=b,wspace=0,hspace=0)
        
    plt.savefig(metapath+"\\"+type+"_error.png")

    if show == True:plt.show()

def deltaE(Ca,Cb):
    [h,s,v]=Ca
    Ca = HSVColor(h,s,v)
    [h,s,v]=Cb
    Cb = HSVColor(h,s,v)
    return delta_e_cie2000(convert_color(Ca, LabColor, target_illuminant='d65'),convert_color(Cb, LabColor, target_illuminant='d65'))

def ave_deltaE(pallets):
    sum = 0
    combi = itertools.combinations(pallets,2)
    pattern = len(pallets)*(len(pallets)-1)/2
    for a,b in combi:
        for n in range(len(a)):
            sum += deltaE(a[n],b[n])/pattern/len(a)
    return sum/69.37387082104067

def deltaE_transition(data,type):
    deltaE=[]
    for pallets,_ in zip(data,tqdm(range(len(data)))):
        deltaE.append(ave_deltaE(convert(pallets,type,target="HSV")))
    return np.array(deltaE)

def est(pallet):
    def HSV_to_Mansell(Ca):
        [h,s,v] = Ca
        return [h,s/100*11,v/100*11]

    def pleasantness(Ca,Cb):
        b =[-0.741028,\
            0.106610 * 10**(-1),\
            -0.251222 * 10**(-2),\
            0.194627 * 10**(-3),\
            0.212212 * 10**(-2),\
            0.369588 * 10**(-2),\
            0.596614 * 10**(-2),\
            0.190872 * 10**(-5),\
            -0.213873 * 10**(-3),\
            -0.378737 * 10**(-5),\
            -0.585471 * 10**(-5),\
            -0.301059 * 10**(-5)]

        def Z(Ca):
            [H,C,V]=Ca
            theta = H/180*np.pi
            Z1 = C*np.cos(theta)
            Z2 = C*np.sin(theta)
            Z3 = 8.33*V
            return [0,Z1,Z2,Z3]

    
        A = Z(Ca)
        B = Z(Cb)
    
        x =  b[0]
        x += b[1]*(A[1]**2+B[1]**2)
        x += b[2]*(A[2]**2+B[2]**2)
        x += b[3]*(A[3]**2+B[3]**2)
        x += b[4]*(A[1]*A[2]+B[1]*B[2])
        x += b[5]*(A[1]*B[1])
        x += b[6]*(A[2]*B[2])
        x += b[7]*(A[3]**3*B[3]**3)
        x += b[8]*(A[1]**2*A[3]+B[1]**2*B[3])
        x += b[9]*(A[3]**2*B[1]+A[1]*B[3]**2)
        x += b[10]*(A[3]**2*B[2]+A[2]*B[3]**2)
        x += b[11]*(A[3]**2*B[3]+A[3]*B[3]**2)

        return x


    def contrast(deltaE,Clist):
        
        return deltaE+3*np.max(list(Clist))

    def floridness(Ca):
        def f(H,S):
            if S == 0:
                return 0
        
            n = int((H+360/18)//9)

            value = [10,10,8,4,2,0,2,6,10]

            return value[n]

        [h,s,v] = Ca
        #f(h,s)+
        return 0.2*(v+0.5)**2

    index = itertools.combinations(range(len(pallet)),2)

    p = [HSV_to_Mansell(c) for c in list(pallet)]

    pleasantness = 0
    deltaElist = []
    Clist = []
    floridness = 0
    
    for a,b in index:
        pleasantness += pleasantness(p[a],p[b])
        deltaElist.append(deltaE(p[a],p[b]))

    for a in range(len(p)):
        Clist.append(p[a][1])
        floridness += floridness(pallet[a])

    ave_deltaE=np.mean(Clist)
    
    contrast = contrast(ave_deltaE,Clist)

    return pleasantness,contrast,floridness

def est_multi(pallets,type):
    A=0
    B=0
    C=0
    pallets = convert(pallets,type,target="HSV")
    n = len(pallets)

    for pallet in pallets:
        a,b,c = est(list(pallet))
        A+=a
        B+=b
        C+=c
    return A/n,B/n,C/n

def est_transition(data,type):
    pleasantness  = []
    contrast = []
    floridness   = []
    
    for pallets,_ in zip(data,tqdm(range(len(data)))):
        
        A,B,C = est_multi(pallets,type)

        pleasantness.append(A)
        contrast.append(B)
        floridness.append(C)

    return np.array(pleasantness)/654323.3994526996,np.array(contrast)/24.498883846273664, np.array(floridness)/4073.3445137012645

def grid_est(multi_dE = False):
    os.chdir(metapath)
    type = list(map(str,metapath.split("\\")))[-1]
    pathes = glob.glob(metapath+"/G_**_D_**/")
    ordersG = set()
    ordersD = set()

    for text in pathes:
        _,g,d = map(str,text.replace("G_",",").replace("_D_",",").replace("\\","").split(","))
        ordersG.add(g)
        ordersD.add(d)

    ordersG = list(ordersG)
    ordersD = list(ordersD)

    ordersG.sort(reverse=True)
    ordersD.sort()

    numG = len(ordersG)
    numD = len(ordersD)

    fig, axs = plt.subplots(numD,numG,figsize=(7,7),sharex = True, sharey = True)        

    if multi_dE:plt.ylim(0,1.99)
    else:plt.ylim(0,2.99)
        
    for order_d,x in zip(ordersD,range(numD)):
        for order_g,y in zip(ordersG,range(numG)):
                
                path = metapath+"/G_"+order_g+"_D_"+order_d

                data=np.load(path+"/pallet.npy")
                [_,_,_,iterstep] = list(np.load(path+"\\metadata.npy")[:,1])
                #kernel = np.ones(10)/10
                iter = np.arange(0,len(data))*iterstep

                if os.path.isfile(path+"/comfortness.npy") and True:
                    pleasantness  = np.load(path+"/comfortness.npy")
                    contrast     = np.load(path+"/conspicuity.npy")
                    floridness   = np.load(path+"/brilliance.npy")

                else:
                    pleasantness, contrast, floridness = est_transition(data,type)
                
                    np.save(path+"/comfortness.npy", pleasantness )
                    np.save(path+"/conspicuity.npy", contrast)
                    np.save(path+"/brilliance.npy", floridness )

                    #pleasantness  = np.convolve(pleasantness,kernel,mode="same")
                    #contrast  = np.convolve(contrast,kernel,mode="same")
                    #floridness   = np.convolve(floridness,kernel,mode="same")

                if multi_dE == True:
                    if os.path.isfile(path+"/deltaE.npy") and False:
                        deltaE = np.load(path+"/deltaE.npy")
                    else:
                        deltaE = deltaE_transition(data,type)
                        #deltaE = np.convolve(deltaE,kernel,mode="same")
                        np.save(path+"/deltaE.npy",deltaE)
                        
                    deltaE = deltaE
                    pleasantness *= deltaE
                    contrast *= deltaE
                    floridness *= deltaE

                axs[x,y].plot(iter,pleasantness  ,label="pleasantness",linestyle="solid",  linewidth=1)
                axs[x,y].plot(iter,contrast ,label="contrast",linestyle='dashdot',linewidth=1)
                axs[x,y].plot(iter,floridness   ,label="floridness",linestyle='dashed', linewidth=1)

                if multi_dE == True:
                    axs[x,y].plot(iter,deltaE,label="D",linestyle="-", linewidth=0.5,alpha=0.5)
                    axs[x,y].set_yticks([0.0, 0.5, 1.0, 1.5])
                else:
                    axs[x,y].set_yticks([0.0,0.5,1.0,1.5,2.0,2.5])

                axs[x,y].axhline(1,linewidth=0.5,linestyle='dashdot',color="black")

                if x == 0 and y == 0:axs[x,y].legend(loc='upper left')

                axs[x,y].tick_params(bottom=False,
                        left=False,
                        right=False,
                        top=False)
    l=0.08
    r=1-l
    b=0.08
    t=1-b

    dx = (1-2*l)/numG
    dy = (1-2*b)/numD

    plt.yticks(rotation='vertical')

    fig.text(0.5, 0.02, 'Iteration', ha='center')

    for lr,n in zip(reversed(ordersD),range(numD)):
        fig.text(r+0.01,b+dy*(1+2*n)/2, 'η_D : 10^('+str(lr)+")", va='center',rotation='vertical')

    for lr,n in zip(ordersG,range(numG)):
        fig.text(l+dx*(1+2*n)/2,t+0.01, 'η_G : 10^('+str(lr)+")", ha='center')

    plt.subplots_adjust(left=l,right=r,top=t,bottom=b,wspace=0,hspace=0)
    
    if multi_dE:
        fig.text(0.005, 0.5, 'Relative predicted value of color affections multiplied by D', va='center', rotation='vertical')
        plt.savefig(metapath+"\\"+type+"_estimation_multidE.png")
    else:
        fig.text(0.005, 0.5, 'Relative predicted value of color affections', va='center', rotation='vertical')
        plt.savefig(metapath+"\\"+type+"_estimation.png")

    if show == True:plt.show()

def pack():
    real = np.zeros((2336,4,3))
    for n in range(2336):
        real[n] = np.loadtxt("data\\col-scheme\\csv\\val ("+str(n+1)+")_rgb.csv",delimiter=",")
    return real

def hist_element(pallets,type):
    data = np.array(convert(list(pallets/255),"RGB",target="HSV"),dtype=int)
    data[:,:,0] += np.array([[0 if sat !=0 else +1000 for sat in pal] for pal in list(data[:,:,1])])

    fig,axs = plt.subplots(3,1)
    fig.tight_layout(rect=[0,0,1,0.90])
    titles = ["Hue","Saturation","Value"]
    for n in range(3):
        if n == 0:
            edges = range(0,361,360//24) 
            num, bins, patches = axs[n].hist(np.array(data[:,:,n]).ravel(),bins=edges)

        if n == 1:
            edges = range(0,101,5)
            num, bins, patches = axs[n].hist(np.array(data[:,:,n]).ravel(),bins=edges)

        if n == 2:
            edges = range(0,101,5)
            num, bins, patches = axs[n].hist(np.array(data[:,:,n]).ravel(),bins=edges)

        axs[n].set_xlabel(titles[n])
        axs[n].set_ylabel("Frequency")

        if n == 0:
            for i in range(24):
                if n == 0:
                    color = plc.hsv_to_rgb([i/24, 1, 1])
                patches[i].set_facecolor(color)
        else:
            for i in range(20):
                if n == 1:
                    color = plc.hsv_to_rgb([1, i/20, 0.8])
                if n == 2:
                    color = plc.hsv_to_rgb([1, 1, i/20])
                patches[i].set_facecolor(color)
                
        print(n,"\n",num)
                
    plt.subplots_adjust(hspace=0.55)
    plt.tight_layout()
    plt.savefig(metapath+ "\\" + type+ "_3elements_dist")
    plt.show()

#realdata = pack()
#np.random.shuffle(realdata)
#hist_element(realdata*255,"RGB")

def grid_hist_element():
    os.chdir(metapath)
    type = list(map(str,metapath.split("\\")))[-1]
    pathes = glob.glob(metapath+"/G_**_D_**/")
    ordersG = set()
    ordersD = set()

    for text in pathes:
        _,g,d = map(str,text.replace("G_",",").replace("_D_",",").replace("\\","").split(","))
        ordersG.add(g)
        ordersD.add(d)

    ordersG = list(ordersG)
    ordersD = list(ordersD)

    ordersG.sort(reverse=True)
    ordersD.sort()

    numG = len(ordersG)
    numD = len(ordersD)
        
    r_list = []
    g_list = []
    b_list = []

    for order_d,x in zip(ordersD,range(numD)):
        for order_g,y in zip(ordersG,range(numG)):
            path = metapath+"/G_"+order_g+"_D_"+order_d
            if float(order_d)>-4 and float(order_g)>-4:
                data = np.array(convert(np.load(path+"/pallet.npy")[-1],type,target="RGB"))
                r_list += list(data[:,:,0])
                g_list += list(data[:,:,1])
                b_list += list(data[:,:,2])

    pallets = np.array([r_list,g_list,b_list]).transpose(1,2,0)

    hist_element(pallets,type)

def grid_hist_pattern():
    os.chdir(metapath)
    type = list(map(str,metapath.split("\\")))[-1]
    pathes = glob.glob(metapath+"/G_**_D_**/")
    ordersG = set()
    ordersD = set()

    for text in pathes:
        _,g,d = map(str,text.replace("G_",",").replace("_D_",",").replace("\\","").split(","))
        ordersG.add(g)
        ordersD.add(d)

    ordersG = list(ordersG)
    ordersD = list(ordersD)

    ordersG.sort(reverse=True)
    ordersD.sort()

    numG = len(ordersG)
    numD = len(ordersD)
    
    dist = []
    label = range(24)
 
    fig = plt.figure() 

    ax = fig.add_subplot(1,1,1)
    dist = []
    for order_d,x in zip(ordersD,range(numD)):
        for order_g,y in zip(ordersG,range(numG)):
            path = metapath+"/G_"+order_g+"_D_"+order_d
            if float(order_d)>-4 and float(order_g)>-4:
                data = np.load(path+"/pallet.npy")
                nepoch,npal,ncol,nchan = np.shape(data)
                for j in range(npal):
                    for k in range(ncol):
                        target = np.argmax(data[0,j,k,0:24])
                        dist.append(list(np.sum(np.roll(np.delete(data[0,j,:,0:24],k,0),-target),axis = 0)))
    sumhue = np.sum(np.array(dist),axis=0)
    patches = ax.bar(label,sumhue/768)
    for i in range(24):
        color = plc.hsv_to_rgb([i/24, 1, 1 ])
        patches[i].set_facecolor(color)
        

    plt.plot(label[0],sumhue[0]/768+0.003,marker="D",markersize = 8,label="Selected hue of anther color",color="black")
    plt.annotate("Comparison Hue", xy=(label[0]+0.8,sumhue[0]/768+0.0019),color="black",backgroundcolor=(1,1,1,0.6))

    plt.ylim(0.48,0.55)
    ax.set_ylabel('Average Output Values')
    ax.set_xlabel('Relative hue indexes')

    plt.savefig(metapath+"\\initHSVpattern.png")
    
    fig = plt.figure() 

    ax = fig.add_subplot(1,1,1)
    dist = []
    for order_d,x in zip(ordersD,range(numD)):
        for order_g,y in zip(ordersG,range(numG)):
            path = metapath+"/G_"+order_g+"_D_"+order_d
            if float(order_d)>-4 and float(order_g)>-4:
                data = np.load(path+"/pallet.npy")
                nepoch,npal,ncol,nchan = np.shape(data)
                for j in range(npal):
                    for k in range(ncol):
                        target = np.argmax(data[-1,j,k,0:24])
                        dist.append(list(np.sum(np.roll(np.delete(data[-1,j,:,0:24],k,0),-target),axis = 0)))
    sumhue = np.sum(np.array(dist),axis=0)
    patches = ax.bar(label,sumhue/768)
    for i in range(24):
        color = plc.hsv_to_rgb([i/24, 1, 1 ])
        patches[i].set_facecolor(color)

    plt.plot(label[0],sumhue[0]/768+0.003,marker="D",markersize = 8,label="Selected hue of anther color",color="black")
    plt.annotate("Comparison Hue", xy=(label[0]+0.8,sumhue[0]/768+0.0019),color="black",backgroundcolor=(1,1,1,0.6))

    plt.ylim(0.48,0.55)
    ax.set_ylabel('Average Output Values')
    ax.set_xlabel('Relative hue indexes')
    plt.savefig(metapath+"\\trainedHSVpattern.png")


show = False

while True:
#    try:
    cm_list = list(map(str,input(">>>").split("&")))
    for cm in cm_list:
        print(cm)
        newpath, newoption = map(str,cm.split(","))
        newpath = newpath.replace('"', '')

        if newoption != "_":option = option=list(map(str,newoption.split(":")))

        if len(option)==2:
            [option,step]=option
            step=int(step)

        else:
            option=option[0]
            step=1

        if newpath == "_":None

        elif option[0] == "g":
            metapath = newpath

        else:
            path = newpath
            [type,lr_G,lr_D,iterstep] = list(np.load(path+"\\metadata.npy")[:,1])

        #model pallet
        if option == "mp":save_all(path,step=step)
        
        #model error
        elif option == "me":
            errG = list(np.load(path+"\\errG.npy")) 
            fig, ax = plt.subplots()
            ax.plot(errG)
            plt.show()

        #grid search pallet
        elif option == "gp":
            N = len(np.load(glob.glob(metapath+"/G_**_D_**/pallet.npy")[1]))

            for n in tqdm(range(0,N,step)):
                grid_img(n,noNum=False,fileindex=n//step)

        #grid search pallet last
        elif option == "gpl":
            n = len(np.load(glob.glob(metapath+"/G_**_D_**/pallet.npy")[1]))
            grid_img(n-1)

        elif option == "gpf":
            n = len(np.load(glob.glob(metapath+"/G_**_D_**/pallet.npy")[1]))
            grid_img(0)

        #grid search error
        elif option == "ger":grid_err()

        #model estimation
        elif option == "me":
            data=np.load(path+"/pallet.npy")
            pleasantness, contrast, floridness = est_transition(data,type)
            iter = np.arange(0,len(data))*iterstep

            plt.plot(iter,pleasantness,label="pleasantness")
            plt.plot(iter,contrast,label="contrast")
            plt.plot(iter,floridness,label="floridness")

            plt.legend()
            if show == True:plt.show()

        #grid search estimation
        elif option == "ges":grid_est()

        #grid search estimation with dE
        elif option == "gesde":grid_est(multi_dE=True)
        
        elif option == "mde":
            data=np.load(path+"/pallet.npy")
            plt.plot(deltaE_transition(data,type))
            if show == True:plt.show()

        elif option == "gh":
            grid_hist_element()

        plt.close()
    print("\n")

#generate all graph]

"""
D:\PythonPrograms\ColorPalletGan\GANandExtractor\Results\RGB,ges&_,gesde&_,ger&_,gpf&_,gh&D:\PythonPrograms\ColorPalletGan\GANandExtractor\Results\HSV,ges&_,gesde&_,ger&_,gpf&_,gh&D:\PythonPrograms\ColorPalletGan\GANandExtractor\Results\SHSV,ges&_,gesde&_,ger&_,gpf&_,gh
"""
"""
D:\PythonPrograms\ColorPalletGan\GANandExtractor\Results\RGB,gh&D:\PythonPrograms\ColorPalletGan\GANandExtractor\Results\HSV,gh&D:\PythonPrograms\ColorPalletGan\GANandExtractor\Results\SHSV,gh
"""