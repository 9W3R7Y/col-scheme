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
import os
import shutil

log_saturation = []
log_value = []

class Env:
    def __init__(self,x,y,lr_D,lr_G):
        self.real_num = 2336
        self.col_num = 4
        self.batch_size=16
        self.epoch = 250
        self.real_loader = Loader.load_real_values("data\\col-scheme\\csv\\",
                                            self.real_num,
                                            self.col_num,
                                            self.batch_size)
        self.GAN = LSGAN(self.real_loader,self.real_num,epoch=self.epoch,batch_size=self.batch_size,nz=16,col_num=self.col_num,x=x,y=y,lr_D=lr_D,lr_G=lr_G)
        
    def Run(self):
        self.GAN.Run(self.batch_size)

#データセットクラスの定義 https://stackoverflow.com/questions/52818145/why-pytorch-dataloader-behaves-differently-on-numpy-array-and-list
class RealDataset(Dataset):
    def __init__(self,data):
        self.data=data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
         

class Loader:           #教師データのロード
    def __init__(self):
        return None

    def load_real_values(value_loc,real_num,col_num,batch_size):
        print("loading dataset...")
        real = np.zeros((real_num,(24+2),col_num))
        for n in range(real_num):
            #print("n =",n)
            real[n] = np.transpose(np.loadtxt(value_loc+"val ("+str(n+1)+")_divided.csv",delimiter=",")) #1次元画像として扱うため 3ch * 8pixcel
        ##################################333
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        real = torch.tensor(real*2-1,dtype=torch.float32)   #real_data を doubleに変換
        real_loader = DataLoader(RealDataset(real),batch_size=batch_size,shuffle = True,worker_init_fn=worker_init_fn)
        print("complete")
        return real_loader

class Generator(nn.Module):
    def __init__(self,nz):
        super(Generator,self).__init__()

        self.layers = nn.ModuleList([  #GeneratorのCNN ノイズ：32*1 => カラーパレット：3*8
            nn.Sequential(
                nn.Linear(nz,4*6,bias=True),
                nn.BatchNorm1d(4*6),
                nn.Dropout(),
                #nn.LeakyReLU(negative_slope = 0.2,inplace = True),
                nn.PReLU(),
                ),
            nn.Sequential(
                nn.Linear(4*6,4*12,bias=True),
                nn.BatchNorm1d(4*12),
                nn.Dropout(),
                #nn.LeakyReLU(negative_slope = 0.2,inplace = True),
                nn.PReLU(),
                ),
            nn.Sequential(
                nn.Linear(4*12,4*26,bias=True),
                nn.Dropout(),##############
                nn.Tanh(), #彩度が低くなるのこれのせいか？違いましたs
                )
            ])

        #torch.save(self.layers.state_dict(), "models/generator_init")


    def forward(self,z):
        #print("G")
        #print(z.size())
        for layer in self.layers:
            z = layer(z)

        #-1 ~ 1 に正規化


        z = z.view(-1,26*4)

        return z

class Discriminator(nn.Module):

    def __init__(self,col_num):
        super(Discriminator,self).__init__()
        self.col_num = col_num

        self.layers = nn.ModuleList([  #GeneratorのCNN ノイズ：32*1 => カラーパレット：3*8
            nn.Sequential(
                nn.Linear((24+2)*4,4*12),
                nn.BatchNorm1d(4*12),
                nn.Dropout(),
                #nn.LeakyReLU(negative_slope = 0.2,inplace = True),
                nn.PReLU(),
                ),
            nn.Sequential(
                nn.Linear(4*12,4*6),
                nn.BatchNorm1d(4*6),
                nn.Dropout(),
                #nn.LeakyReLU(negative_slope = 0.2,inplace = True),
                nn.PReLU(),
                ),
            nn.Sequential(
                nn.Linear(4*6,1),
                nn.Sigmoid(), #彩度が低くなるのこれのせいか？違いましたs
                )
            ])

    def forward(self,z):
        for layer in self.layers:
            z = layer(z)
        return z


class LSGAN():
    def __init__(self,real_loader,real_num,epoch,batch_size,nz,col_num,x,y,lr_D,lr_G):
        self.real_loader = real_loader
        self.epoch=epoch
        self.batch_size=batch_size
        self.real_num = real_num
        self.nz = nz
        self.col_num = col_num
        self.Loader = Loader()
        self.criterion = nn.MSELoss()

        self.D = Discriminator(col_num).float()
        self.G = Generator(nz=nz).float()

        self.optimD = optim.Adam(self.D.parameters(),lr=lr_D)
        self.optimG = optim.Adam(self.G.parameters(),lr=lr_G)

        print(self.optimD,self.optimG)

        self.lr_D=lr_D
        self.lr_G=lr_G

        self.iterstep = 100

        self.fixed_noise = torch.randn(testnum,nz,1,1)

        #ディレクトリ作成
        self.pathtry = path+"G_"+str(math.log10(self.lr_G))+"_D_"+str(math.log10(self.lr_D))+"\\"
        os.makedirs(self.pathtry,exist_ok=True)

    def Run(self,batch_size):
        self.D.eval()
        self.G.eval()
        output_log=[]#出力の記録

        fake_data_test = self.G(self.fixed_noise.view(-1,self.nz))

        fake_color_pallet = np.clip(((fake_data_test.view(-1,(24+2),self.col_num)+1).permute(0,2,1)*0.5).view(-1,self.col_num,(24+2)).detach().numpy(),0,1)

        output_log.append(fake_color_pallet)

        #誤差の記録
        errDreal_log = []
        errDfake_log = []
        errG_log = []

        numepoch = 0
        iter = 0

        for epoch in range(self.epoch):
            
            #確認用
            numepoch += 1

            print("epoch", numepoch)

            for n,real_data in enumerate(self.real_loader):#各バッチ枚の処理

                iter += 1
                
                self.D.train()
                self.G.train()

                #print("\tbatch",n)
                #正解データのsizeをNNのinputに合わせる
                real_data = real_data.view(batch_size,(24+2)*4) #batchsize*pixcel*channel

                #Generator入力ノイズの生成

                noise = torch.randn(self.batch_size,self.nz,1)

                #正解データをランダムにしてみる　https://qiita.com/underfitting/items/a0cbb035568dea33b2d7
                real_target = torch.full((self.batch_size,),1.) #realの画像のラベル = 1　にしたい
                fake_target = torch.full((self.batch_size,),0.) #fakeの画像のラベル = 0　にしたい

                ###################
                #Discriminatorの更新
                ###################
                self.D.zero_grad()

                #試行・本物
                output = self.D(real_data)

                errD_real = self.criterion(output, real_target)     #誤差計算

                #偽配色の生成
                fake_data = self.G(noise.view(batch_size,self.nz)).view(batch_size,(24+2)*4)#batchsize*pixcel*channel

                #試行・偽物
                output = self.D(fake_data)
                errD_fake = self.criterion(output, fake_target)     #誤差計算
                
                #Discriminatorの誤差の合計値
                errD = errD_real + errD_fake 
                errD.backward()
                self.optimD.step()

                ################
                #Generatorの更新
                ###############
                
                #Discriminatorに偽物を判断させ、本物と誤認させたい
                fake_data = self.G(noise.view(batch_size,self.nz)).view(batch_size,(24+2)*4)#batchsize*pixcel*channel
                self.G.zero_grad()
                output = self.D(fake_data)

                errG = self.criterion(output, real_target)
                errG.backward()
                self.optimG.step()

                #誤差の記録
                errDreal_log.append(float(errD_real))
                errDfake_log.append(float(errD_fake))
                errG_log.append(float(errG))

                if (iter)%self.iterstep == 0:
                    self.D.eval()
                    self.G.eval()

                    fake_data_test = self.G(self.fixed_noise.view(-1,self.nz))

                    fake_color_pallet = np.clip(((fake_data_test.view(-1,(24+2),self.col_num)+1).permute(0,2,1)*0.5).view(-1,self.col_num,(24+2)).detach().numpy(),0,1)

                    output_log.append(fake_color_pallet)

                    #誤差の記録を可視化
        
        #移動平均をとる
        kernel = np.ones(9)/9

        errDreal_log = np.convolve(np.array(errDreal_log),kernel,mode='same')
        errDfake_log = np.convolve(np.array(errDfake_log),kernel,mode='same')
        errG_log     = np.convolve(np.array(errG_log),kernel,mode='same')
        
        np.save(self.pathtry+'pallet', np.array(output_log))
        np.save(self.pathtry+'errDreal', np.array(errDreal_log))
        np.save(self.pathtry+'errDfake', np.array(errDfake_log))
        np.save(self.pathtry+'errG', np.array(errG_log))
        np.save(self.pathtry+'metadata', np.array([["type",2],["lr_G",self.lr_G],["lr_D",self.lr_D],["iterstep",self.iterstep]],dtype=object))
        
#実験設定用変数

VisualizeEverytime = False
PresentableColorPallet = False
SaveImage = True

ordersD = [-4.5,-4,-3.5,-3]
ordersG = [-4.5,-4,-3.5,-3]

ordersD=list(reversed(ordersD))
testnum = 16

path = "data\\SHSV\\"
try:
    shutil.rmtree(path)
except:
    pass

numD = len(ordersD)
numG = len(ordersG)

for order_d,x in zip(ordersD,range(numD)):
    for order_g,y in zip(ordersG,range(numG)):
        torch.manual_seed(1)

        lr_D = 10**(order_d)
        lr_G = 10**(order_g)
        environment = Env(x,y,lr_D,lr_G)
        environment.Run()


def beep(freq, dur=100):
    import winsound
    winsound.Beep(freq, dur)

for n in range(6):
    beep(2000, 500)
    beep(4000, 500)

