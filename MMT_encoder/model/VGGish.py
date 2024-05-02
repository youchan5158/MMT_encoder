import torch

def get_vggish():
    model_vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model_vggish.eval()

    return model_vggish