### imports ###
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image,ImageChops
import argparse

### defines ###
dtype = torch.cuda.FloatTensor
n_layers = 28

### classes ###
class VGG(nn.Module):
    def __init__(self,n_layers):
        super(VGG,self).__init__()
        self.features = nn.Sequential(
            *list(models.vgg19(pretrained=True).features.children())[0:n_layers])
    def forward(self,x):
        x = self.features(x)
        return x

class Shifter(nn.Module):
    def __init__(self):
        super(Shifter,self).__init__()
        self.shifter = nn.Conv2d(3,3,kernel_size=1,stride=1,padding=0,groups=3)
        self.shifter.weight.data = torch.Tensor([0.229,0.224,0.225]).view(3,1,1,1) #rgb
        self.shifter.bias.data = torch.Tensor([0.485,0.456,0.406])
    def forward(self,x):
        x = self.shifter(x)
        return x

### functions ###
def image_loader(image_name):
    image = Image.open(image_name)
    return image

def image_saver(image,image_name):
    name2save = image_name.replace('.'+image_name.split('.')[-1],'_dp.png')
    image.save(name2save)

def imshow(image,title=None):
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)

def reset_grads(net,require_grad=True):
    for p in net.parameters():
        p.requires_grad = require_grad
    return net

def init_net(n_layers=28):
    shifter = Shifter().cuda()
    shifter = reset_grads(shifter,require_grad=False)
    cnn = VGG(n_layers=n_layers).cuda()
    cnn = reset_grads(cnn,require_grad=False)
    shifter = reset_grads(shifter,require_grad=False)
    return cnn,shifter

def make_step(image,cnn,shifter,n_iters,lr):
    loader = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    unloader = transforms.Compose([transforms.ToPILImage()])
    image = loader(image)
    image = image.unsqueeze(0)
    x = Variable(image.cuda(),requires_grad=True)
    optimizer = torch.optim.Adam([x],lr)
    for i in range(n_iters):
        o = cnn(x)
        loss = -o.norm()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    image = torch.clamp(shifter(x),min=0,max=1)
    image = image.data.view(3,224,224)
    image = unloader(image.cpu())
    return image

def deep_dream(image,cnn,shifter,n_iters=100,lr=0.015,octave_scale=1.5,n_octaves=1):
    o_image = image
    for i in range(n_octaves):
        scale = (i+1)*octave_scale
        if (min(image.size[0]/scale,image.size[1]/scale)<1):
            continue
        else:
            size = (int(image.size[0]/scale),int(image.size[1]/scale))
        d_image = image.resize(size,Image.ANTIALIAS)
        d_image = make_step(d_image,cnn,shifter,n_iters,lr)
        d_image = d_image.resize((image.size[0],image.size[1]),Image.ANTIALIAS)
        o_image = ImageChops.blend(o_image,d_image,0.6)
    return o_image

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',help='path to image',default='./images/me.jpg')
    parser.add_argument('--plot',help='plot image',default=False,action='store_true')
    parser.add_argument('--n_iters',type=int,help='number of gd iterations',default=100)
    parser.add_argument('--lr',type=float,help='learning rate',default=0.015)
    parser.add_argument('--octave_scale',type=float,help='octave scale',default=1.5)
    parser.add_argument('--n_octaves',type=int,help='number of octaves',default=2)
    parser.add_argument('--n_layers',type=int,help='vgg deep index',default=28)
    opt = parser.parse_args()
    return opt

### main ###
opt = get_arguments()
image = image_loader(image_name=opt.image)
cnn,shifter = init_net(opt.n_layers)
image = deep_dream(image,cnn,shifter,n_iters=opt.n_iters,lr=opt.lr,octave_scale=opt.octave_scale,n_octaves=opt.n_octaves)
if (opt.plot):
    imshow(image,title=None)
    plt.show()
image_saver(image,image_name=opt.image)