import os
import numpy as np
import cv2

from model.gan.models import *
from torch.autograd import Variable
import torch.nn as nn
import torch

# def main():
#     property_root = '/home/bong04/data/dyetec_fabric/drape_data/property'
#     file_path = []
#     for (path, dir, files) in os.walk(property_root):
#         for file in files:
#             ext = os.path.splitext(file)[-1].lower()
#             formats = ['.npy']
#             if ext in formats:
#                 img_path = os.path.join(path, file)
#                 file_path.append(img_path)
#     for path in file_path:
#         fabric_property = np.load(path)
#         if fabric_property[1] in ['woven', 'Woven']:  # ['Knit' 'Woven' 'kint' 'knit' 'woven']
#             fabric_property[1] = 0
#         else:
#             fabric_property[1] = 1
#         fabric_property = np.append(fabric_property[1:2], fabric_property[3:])  # name, size 200 제외
#         fabric_property = np.array(fabric_property).astype('float')
#         # print(fabric_property)

#         run(fabric_property)

def run(fabric_property):
    '''
    :param fabric_property: 섬유 물성데이터 입력값 (길이 37인 list)
    :return: byte 형태 이미지
    '''

    input_shape = (1, 224, 224)
    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = Generator()
    G_BA = GeneratorResNet(input_shape, 9)

    if cuda:
        generator.cuda()
        G_BA = G_BA.cuda()

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # B image
    checkpoint = torch.load('./model/gan/399_generator.pth', map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()

    checkpoint = torch.load('./model/gan/G_BA_38.pth', map_location=torch.device('cpu'))
    G_BA.load_state_dict(checkpoint)
    G_BA.eval()

    fabric_property = np.array(fabric_property).reshape((1, -1))
    properties = Variable(torch.tensor(fabric_property).type(FloatTensor))

    b_c_gen_imgs = generator(properties)
    b_cycle_gen_imgs = G_BA(b_c_gen_imgs)

    b_cycle_gen_img = b_cycle_gen_imgs[0]
    b_cycle_gen_img = b_cycle_gen_img.cpu().detach().numpy().transpose(1, 2, 0)
    b_cycle_gen_img = cv2.resize(b_cycle_gen_img, (320, 320))
    b_img_str = cv2.imencode('.png', b_cycle_gen_img*255)[1].tostring()

    # F image
    checkpoint = torch.load('./model/gan/F_260_generator.pth', map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()

    f_c_gen_imgs = generator(properties)

    su = np.sum(b_cycle_gen_img < 1)
    if su < 15900:
        checkpoint = torch.load('./model/gan/F_G_BA_140.pth', map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load('./model/gan/F_G_BA_105.pth', map_location=torch.device('cpu'))
    G_BA.load_state_dict(checkpoint)
    G_BA.eval()

    f_cycle_gen_imgs = G_BA(f_c_gen_imgs)

    f_cycle_gen_img = f_cycle_gen_imgs[0]
    f_cycle_gen_img = f_cycle_gen_img.cpu().detach().numpy().transpose(1, 2, 0)
    f_cycle_gen_img = cv2.resize(f_cycle_gen_img, (320, 320))
    f_img_str = cv2.imencode('.png', f_cycle_gen_img*255)[1].tostring()

    # cv2.imshow('b', b_cycle_gen_img)
    # cv2.imshow('f', f_cycle_gen_img)
    # cv2.waitKey(0)

    # print(img_str)
    # nparr = np.fromstring(img_str, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)

    return b_img_str, f_img_str


class Generator(nn.Module):  ## 37 <- num of properties
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(37, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod((1, 224, 224)))),
            nn.Tanh()
        )

    def forward(self, labels):
        img = self.model(labels)
        img = img.view(img.size(0), *(1, 224, 224))
        return img

if __name__ == '__main__':
    main()