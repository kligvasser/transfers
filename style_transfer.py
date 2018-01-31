### imports ###
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import torchvision
import argparse

### defines ###
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor
content_layers_default = ['conv_4']
style_layers_default = ['conv_1','conv_2','conv_3','conv_4','conv_5']

### classes ###
class ContentLoss(nn.Module):
    def __init__(self,target,weight):
        super(ContentLoss,self).__init__()
        self.target = target.detach()*weight
        self.weight = weight
        self.criterion = nn.MSELoss()
    def forward(self,input):
        self.loss = self.criterion(input*self.weight,self.target)
        self.output = input
        return self.output
    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class GramMatrix(nn.Module):
    def forward(self,input):
        a,b,c,d = input.size()
        features = input.view(a*b,c*d)  # resise F_XL into \hat F_XL
        G = torch.mm(features,features.t())  # compute the gram product
        return G.div(a*b*c*d)

class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss,self).__init__()
        self.target = target.detach()*weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
    def forward(self,input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G,self.target)
        return self.output
    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

### functions ###
def image_loader(image_name,imsize=512):
    loader = transforms.Compose([transforms.Resize(imsize),transforms.CenterCrop(imsize),transforms.ToTensor()])
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image

def image_saver(image,image_name):
    name2save = image_name.replace('.'+image_name.split('.')[-1],'_st.png')
    torchvision.utils.save_image(image,name2save)

def imshow(tensor,title=None,imsize=512):
    unloader = transforms.Compose([transforms.ToPILImage()])
    image = tensor.clone().cpu()
    image = image.view(3,imsize,imsize)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)

def get_style_model_and_losses(cnn,style_img,content_img,style_weight=1000,content_weight=1,content_layers=content_layers_default,style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses = []
    model = nn.Sequential()
    gram = GramMatrix()
    model = model.cuda()
    gram = gram.cuda()
    i = 1
    for layer in list(cnn):
        if isinstance(layer,nn.Conv2d):
            name = "conv_"+str(i)
            model.add_module(name,layer)
            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target,content_weight)
                model.add_module("content_loss_"+str(i),content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram,style_weight)
                model.add_module("style_loss_"+str(i),style_loss)
                style_losses.append(style_loss)
        if isinstance(layer,nn.ReLU):
            name = "relu_"+str(i)
            model.add_module(name,layer)
            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_"+str(i),content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
            i += 1
        if isinstance(layer,nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name,layer)
    return model,style_losses,content_losses

def get_input_param_optimizer(input_img):
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param,optimizer

def run_style_transfer(cnn,content_img,style_img,input_img,n_steps=100,style_weight=1000,content_weight=1):
    print('Building the style transfer model..')
    model,style_losses,content_losses = get_style_model_and_losses(cnn,style_img,content_img,style_weight,content_weight)
    input_param,optimizer = get_input_param_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= n_steps:
        def closure():
            input_param.data.clamp_(0,1)
            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()
            run[0] += 1
            if run[0]%50 == 0:
                print('{} Style Loss : {:4f} Content Loss: {:4f}'.format(run,style_score.data[0],content_score.data[0]))
            return style_score+content_score
        optimizer.step(closure)
    input_param.data.clamp_(0,1)
    return input_param.data

def init_net():
    cnn = models.vgg19(pretrained=True).features
    cnn.cuda()
    return cnn

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_image',help='path to image',default='./images/starry_night.jpg')
    parser.add_argument('--contenet_image',help='path to image',default='./images/me.jpg')
    parser.add_argument('--plot',help='show image',default=False,action='store_true')
    parser.add_argument('--imsize',type=int,help='image size',default=512)
    parser.add_argument('--n_steps',type=int,help='number of steps',default=100)
    parser.add_argument('--style_weight',type=int,help='style weight',default=1000)
    parser.add_argument('--content_weight',type=int,help='content weight',default=1)
    opt = parser.parse_args()
    return opt

### main ###
opt = get_arguments()
style_img = image_loader(opt.style_image,opt.imsize).type(dtype)
content_img = image_loader(opt.contenet_image,opt.imsize).type(dtype)
cnn = init_net()
input_img = content_img.clone()
output = run_style_transfer(cnn,content_img,style_img,input_img,n_steps=opt.n_steps,style_weight=opt.style_weight,content_weight=opt.content_weight)
if (opt.plot):
    imshow(output,title='Output Image',imsize=opt.imsize)
    plt.show()
image_saver(output,opt.contenet_image)