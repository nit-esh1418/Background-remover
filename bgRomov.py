import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
from data_loader_cache import normalize, im_reader, im_preprocess
from models import ISNetDIS

import logging
logging.basicConfig(level=logging.INFO)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

class GOSNormalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image

transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])

def load_image(im_path, hypar):
    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0) 

def build_model(hypar, device):
    net = hypar["model"]

    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if hypar["restore_model"] != "":
        net.load_state_dict(torch.load(os.path.join(hypar["model_path"], hypar["restore_model"]), map_location=device))
        net.to(device)
    net.eval()
    return net

def predict(net, inputs_val, shapes_val, hypar, device):
    net.eval()

    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)
    ds_val = net(inputs_val_v)[0]

    pred_val = ds_val[0][0, :, :, :]
    pred_val = torch.squeeze(F.interpolate(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)

hypar = {}
hypar["model_path"] = "./saved_models"
hypar["restore_model"] = "isnet.pth"
hypar["interm_sup"] = False
hypar["model_digit"] = "full"
hypar["seed"] = 0
hypar["cache_size"] = [1024, 1024]
hypar["input_size"] = [1024, 1024]
hypar["crop_size"] = [1024, 1024]
hypar["model"] = ISNetDIS()


net = build_model(hypar, device)

def remove_background(image_path, output_path):
    logging.info(f"Removing background from {image_path}")
    logging.info(f"Output path: {output_path}")

    image_tensor, orig_size = load_image(image_path, hypar)
    mask = predict(net, image_tensor, orig_size, hypar, device)

    pil_mask = Image.fromarray(mask).convert('L')
    im_rgb = Image.open(image_path).convert("RGB")

    im_rgba = im_rgb.copy()
    im_rgba.putalpha(pil_mask)
    # im_rgba.save(output_path)
    try:
        im_rgba.save(output_path)
        logging.info(f"Background removed image saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving image: {e}")


# image_path = '14.jpg'
# output_path = 'output4.png'
# remove_background(image_path, output_path)
# print(f"Background removed image saved to {output_path}")
