from pathlib import Path
import torch
import torch.nn as nn
from timesformer import TimeSformer
from video_compression import video_compression


class MMT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(MMT, self).__init__()

        self.model_if = TimeSformer(img_size=img_size, num_classes=num_classes, num_frames=num_frames, attention_type=attention_type, pretrained_model=pretrained_model, **kwargs)
        self.model_if.cuda()

        self.model_mv = TimeSformer(img_size=img_size, num_classes=num_classes, in_chans=2 ,num_frames=num_frames-1, attention_type=attention_type, pretrained_model=pretrained_model, **kwargs)
        self.model_mv.cuda()

        self.model_rf = TimeSformer(img_size=img_size, num_classes=num_classes, num_frames=num_frames-1, attention_type=attention_type, pretrained_model=pretrained_model, **kwargs)
        self.model_rf.cuda()

        self.head = nn.Linear(768 * 3, num_classes) if num_classes > 0 else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward_feature(self, x):
        outputs = []
        
        all_flow_tensors, all_residual_tensors = video_compression(x)
        #

        # output_dim : embed_dim
        output_if = self.model_if(x)        # batch x embed_dim(768)
        output_mv = self.model_mv(all_flow_tensors)
        output_rf = self.model_rf(all_residual_tensors)
        

        outputs.append(output_if)
        outputs.append(output_mv)        
        outputs.append(output_rf)

        concatenated_features = torch.cat(outputs, dim=1)

        return concatenated_features


    def forward(self, x):   # (batch x channels x frames x height x width)
        x = self.forward_feature(x)
        x = self.dropout(x)
        x = self.head(x)
        return x