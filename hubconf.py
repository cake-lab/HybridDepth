import torch
from model.main import DepthNetModule


def HybridDepth_Nyu5(pretrained=False, **kwargs):
    # Replace with your model class
    model = DepthNetModule(**kwargs)
    if pretrained:
        pretrained_resource = "url::https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NYUBest5-DFV-Trained.ckpt"
        # Load pretrained weights
        state_dict = torch.hub.load_state_dict_from_url(pretrained_resource)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        pretrained_resource = None
        
    return model


def HybridDepth_Nyu10(pretrained=False, **kwargs):
    # Replace with your model class
    model = DepthNetModule(**kwargs)
    if pretrained:
        pretrained_resource = "url::https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NYUBest10-DFV-Trained.ckpt"
        # Load pretrained weights
        state_dict = torch.hub.load_state_dict_from_url(pretrained_resource)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        pretrained_resource = None
        
    return model


def HybridDepth_DDFF5(pretrained=False, **kwargs):
    # Replace with your model class
    model = DepthNetModule(**kwargs)
    if pretrained:
        pretrained_resource = "url::https://github.com/cake-lab/HybridDepth/releases/download/v2.0/DDFF12.ckpt"
        # Load pretrained weights
        state_dict = torch.hub.load_state_dict_from_url(pretrained_resource)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        pretrained_resource = None
        
    return model


def HybridDepth_Nyu_pretrainedDFV(pretrained=False, **kwargs):
    # Replace with your model class
    model = DepthNetModule(**kwargs)
    if pretrained:
        pretrained_resource = "url::https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NyuBest5.ckpt"
        # Load pretrained weights
        state_dict = torch.hub.load_state_dict_from_url(pretrained_resource)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        pretrained_resource = None
        
    return model