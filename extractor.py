import colorsys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt
import colormath
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from tqdm import tqdm
import glob
class Env():
    def __init__(self):
        self.CPE = ColorPalletextractor()
    def Run(self):
        source_loc  = "data\col-scheme\source\*"
        img_loc     = "data\col-scheme\img\\"
        csv_loc    = "data\col-scheme\csv\\"
        output_num = 0
        for current in tqdm(glob.glob(source_loc)):
            try:
                sigma_b = 0.01
                sigma_c = 100
                self.CPE.Run(   sigma_b,
                                sigma_c,
					            current,
					            csv_loc+"val ("+str(output_num+1)+")",
					            img_loc+"img ("+str(output_num+1)+").jpg")
                output_num += 1
            except:
                import traceback
                traceback.print_exc()
                pass

class ColorPalletextractor():
    def __init__(self):
        self.col_red_lev = 32                          
        self.tone = int(np.round(256/self.col_red_lev))
        self.color_num = 8                             
        self.color_num_fin = 4

    def load_img(self,img_source):
        img_pil = Image.open(img_source).convert("RGB")
        img_resized = img_pil.resize((200,200))      
        img = np.array(img_resized).transpose(1,0,2) 
        width,height = img_resized.size              
        return [img,img_pil,width,height]

    def make_distro(self,img,width,height):
        distro = np.zeros((self.tone,self.tone,self.tone))
        distro_full = np.zeros((256,256,256)) 

        for x in range(width):
                for y in range(height):
                        [r,g,b] = img[x,y]
                        distro_full[r,g,b] += 1
                        r = int(r/self.col_red_lev)
                        g = int(g/self.col_red_lev) 
                        b = int(b/self.col_red_lev)
                        distro[r,g,b] += 1 
        return[distro,distro_full]

    def make_gauss(self,gauss_r,sigma):
        gauss = np.zeros((2*gauss_r+1,2*gauss_r+1,2*gauss_r+1))
        gauss_sum = 0
        for x in range(2*gauss_r+1):                                            
                for y in range(2*gauss_r+1):
                        for z in range(2*gauss_r+1):
                                d_x = x - gauss_r
                                d_y = y - gauss_r
                                d_z = z - gauss_r
                                r_2 = d_x**2+d_y**2+d_z**2                      
                                gauss[x,y,z]=np.exp((-1*r_2)/(2*(sigma**2)))   
                                gauss_sum += gauss[x,y,z]            
        return [gauss,gauss_r,gauss_sum]

    def blur(self,distro,gauss,gauss_r,gauss_sum):
        distro_blurred= np.zeros((self.tone,self.tone,self.tone))        
        for R in range(self.tone):
            for G in range(self.tone):
                for B in range(self.tone):
                    amount = 0
                    gauss_out = 0
                    for x in range(2*gauss_r+1):
                        for y in range(2*gauss_r+1):
                            for z in range(2*gauss_r+1):
                                if_r_inside = 0 <= R-gauss_r+x and R-gauss_r+x < self.tone
                                if_g_inside = 0 <= G-gauss_r+y and G-gauss_r+y < self.tone
                                if_b_inside = 0 <= B-gauss_r+z and B-gauss_r+z < self.tone
                                if if_r_inside and if_g_inside and if_b_inside:
                                    amount += distro[R-gauss_r+x,G-gauss_r+y,B-gauss_r+z]*gauss[x,y,z]
                                else:
                                    gauss_out += gauss[x,y,z]
                    amount*=gauss_sum/(gauss_sum-gauss_out)
                    distro_blurred[R,G,B]=amount

        return [distro_blurred]

    def get_dim_index(self,x,n):
            b = x%n
            x = x//n
            g = x%n
            x = x//n
            r = x
            return [int(r),int(g),int(b)]

    def extract(self,distro,distro_full,gauss,gauss_r):
        col_scheme = np.zeros((self.color_num,3))
        for n in range(self.color_num): 
            x = np.argmax(distro)
            [r,g,b] = self.get_dim_index(x,256/self.col_red_lev)

            distro_full_in_range =\
                distro_full[r*self.col_red_lev:(r+1)*self.col_red_lev,\
                            g*self.col_red_lev:(g+1)*self.col_red_lev,\
                            b*self.col_red_lev:(b+1)*self.col_red_lev]

            x_full = np.argmax(distro_full_in_range)
            [r_full,g_full,b_full] = self.get_dim_index(x_full,self.col_red_lev)
            col_scheme[n] =  [int(r*self.col_red_lev+r_full),\
                                int(g*self.col_red_lev+g_full),\
                                int(b*self.col_red_lev+b_full)]

            for x in range(2*gauss_r+1):
                if 0 <= r-x+gauss_r and r-x+gauss_r < self.tone:
                    for y in range(2*gauss_r+1):
                        if 0 <= g-y+gauss_r and g-y+gauss_r < self.tone:
                            for z in range(2*gauss_r+1):
                                if 0 <= b-z+gauss_r and b-z+gauss_r < self.tone:
                                    distro[r-x+gauss_r,\
                                           g-y+gauss_r,\
                                           b-z+gauss_r] *= 1-gauss[x,y,z]
        return [col_scheme]   
    
    def sat(self,color):
        M = np.max(color)
        m = np.min(color)
        if M == m:
            return 0
        else:
            return ((M - m)/M)
    def val(self,color):
        return (np.max(color) / 255)
    def hue(self,color):
        if color[0]==color[1]==color[2]:
            return 0
        max_channel = np.argmax(color)
        max = np.max(color)
        min = np.min(color)
        if max_channel == 0:
            h = 60 * (color[1]-color[2]) / (max - min)
        elif max_channel == 1:
            h = 60 * (color[2]-color[0]) / (max - min) + 120
        elif max_channel == 2:
            h = 60 * (color[0]-color[1]) / (max - min) + 240
        if h < 0:
            h += 360
        return (h/360)

    def HSV_from_RGB(self,color):
        return [self.hue(color),self.sat(color),self.val(color)]

    def LAB_from_RGB(self,color):
        return convert_color(sRGBColor(*(color / 255)), LabColor, target_illuminant='d65')

    def delta_E(self,A,B):
        delta = delta_e_cie2000(self.LAB_from_RGB(A), self.LAB_from_RGB(B))
        return delta

    def export(self,rgb_scheme,hsv_scheme,shsv_scheme,csv_loc):
        np.savetxt(csv_loc+"_rgb.csv",rgb_scheme,delimiter=",")  
        np.savetxt(csv_loc+"_hsv.csv",hsv_scheme,delimiter=",")  
        np.savetxt(csv_loc+"_shsv.csv",shsv_scheme,delimiter=",")    

    def visualize(self,col_scheme,color_num,img_pil,name,HSV = False):
        if HSV:
            n = 0
            for color in col_scheme:
                [r,g,b] = colorsys.hsv_to_rgb(color[0],color[1],color[2])
                col_scheme[n] = [int(r*255),int(g*255),int(b*255)]
                n += 1
        
        col_scheme_size = 250
        text_area = 80
        margin = 20

        [width,height]=img_pil.size
        rate = col_scheme_size*color_num/width
        width_fit = int(width*rate)
        height_fit = int(height*rate)
        img_pil = img_pil.resize((width_fit,height_fit))

        col_scheme_img = Image.new('RGB', (col_scheme_size*color_num+2*margin,col_scheme_size+text_area+3*margin+height_fit), (255, 255, 255))
        draw = ImageDraw.Draw(col_scheme_img)

        draw.font = ImageFont.truetype("C:\Windows\Fonts\meiryo.ttc", int(text_area*0.6))

        for n in range(color_num):
                r = int(col_scheme[n,0])
                g = int(col_scheme[n,1])
                b = int(col_scheme[n,2])

                draw.rectangle((col_scheme_size*n+margin,2*margin+height_fit,col_scheme_size*(n+1)+margin,col_scheme_size+2*margin+height_fit), fill=(r,g,b))

                r_16 = str(format(r,'x')).zfill(2)
                g_16 = str(format(g,'x')).zfill(2)
                b_16 = str(format(b,'x')).zfill(2)

                code = "#"+r_16+g_16+b_16
                (w,h) = draw.textsize(code)
                draw.text((margin+col_scheme_size*(1/2+n)-(w/2), 2*margin+height_fit+col_scheme_size+(text_area/2)-(h/2)), code, fill=(0,0,0))

        col_scheme_img.paste(img_pil,(margin,margin))

        col_scheme_img.save(name)

    def delete(self,col_scheme,num,num_fin,distro_full):
        n = num
        col_scheme = list(col_scheme)
        
        w_b = []
        for i in range(n):
            if col_scheme[i][0] <= 35 \
                and col_scheme[i][1] <= 35 \
                and col_scheme[i][2] <= 35:
                w_b.append(i)
            if col_scheme[i][0] >= 230 and \
                col_scheme[i][1] >= 230 and \
                col_scheme[i][2] >= 230:
                w_b.append(i)

        for i in sorted(w_b, reverse=True):
            col_scheme.pop(i)
            n-=1

        for i in range(n-num_fin):
            min = [10000,-1,-1]
            for a in range(n):
                for b in range(a+1,n):
                    d = self.delta_E(col_scheme[a],col_scheme[b])
                    if min[0] >= d:
                        min = [d,a,b]
                         
            a_R,a_G,a_B = col_scheme[min[1]]
            b_R,b_G,b_B = col_scheme[min[2]]

            a_content = (distro_full[int(a_R)][int(a_G)][int(a_B)])
            b_content = (distro_full[int(b_R)][int(b_G)][int(b_B)])

            if a_content >= b_content:
                col_scheme.pop(min[2])
            else:
                col_scheme.pop(min[1])

            n -= 1

        return np.array(col_scheme)

    def selective_HSV(self,col_scheme,level = 24):
        normal = col_scheme.copy()
        shsv = np.full((self.color_num_fin,level+2),0)

        for n in range(len(col_scheme)):
            hue_level = (col_scheme[n,0]+1/level/2)//(1/level)%level
            hue = hue_level/level
            if col_scheme[n,1] != 0:
                shsv[n,int(hue_level)] = 1
            shsv[n,-2]=col_scheme[n,1]
            shsv[n,-1]=col_scheme[n,2]
            normal[n,0] = hue

        return(normal,shsv)

    def Run(self,sigma_b,sigma_c,img_source,csv_loc,name):
        
        [img,img_pil,width,height] = self.load_img(img_source)

        [distro,distro_full] = self.make_distro(img,width,height)

        [gauss,gauss_r,gauss_sum] = self.make_gauss(self.tone,sigma_b)

        [distro_blurred] = self.blur(distro,gauss,gauss_r,gauss_sum)

        [gauss,gauss_r,gauss_sum] = self.make_gauss(self.tone,sigma_c)

        [col_scheme] = self.extract(distro_blurred,distro_full,gauss,gauss_r)

        col_scheme = self.delete(col_scheme,self.color_num,self.color_num_fin,distro_full)

        rgb_scheme = col_scheme/255

        for color,n in zip(col_scheme,range(self.color_num_fin)):
            col_scheme[n] = self.HSV_from_RGB(color)

        [hsv_scheme,shsv_scheme] = self.selective_HSV(col_scheme)

        self.visualize(hsv_scheme,self.color_num_fin,img_pil,name,HSV=True)

        self.export(rgb_scheme,hsv_scheme,shsv_scheme,csv_loc)
Env = Env()

Env.Run()