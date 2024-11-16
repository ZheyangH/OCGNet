# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.darknet import *

import torchvision.models as models

class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        # Early residual feature
        x1 = x
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        return x, x1

class CrossViewFusionModule(nn.Module):
    def __init__(self):
        super(CrossViewFusionModule, self).__init__()

    # normlized global_query:B, D
    # normlized value: B, D, H, W
    def forward(self, global_query, value):
        global_query = F.normalize(global_query, p=2, dim=-1)
        value = F.normalize(value, p=2, dim=1)

        B, D, W, H = value.shape
        new_value = value.permute(0, 2, 3, 1).view(B, W*H, D)
        score = torch.bmm(global_query.view(B, 1, D), new_value.transpose(1,2))
        score = score.view(B, W*H)
        with torch.no_grad():
            score_np = score.clone().detach().cpu().numpy()
            max_score, min_score = score_np.max(axis=1), score_np.min(axis=1)
        
        attn = Variable(torch.zeros(B, H*W).cuda())
        for ii in range(B):
            attn[ii, :] = (score[ii] - min_score[ii]) / (max_score[ii] - min_score[ii])
        
        attn = attn.view(B, 1, W, H)
        context = attn * value
        return context, attn

class QueryReferenceFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(QueryReferenceFusion, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.out = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, x1, x2):
        B, D, H, W = x1.shape
        
        if x1.shape != x2.shape:
            x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=False)
                
        x1_flat = x1.view(B, D, H * W).permute(0, 2, 1)  # (B, H*W, D)
        x2_flat = x2.view(B, D, H * W).permute(0, 2, 1)  # (B, H*W, D)
        
        q = self.query(x1_flat)  # (B, H*W, D)
        k = self.key(x2_flat)    # (B, H*W, D)
        v = self.value(x2_flat)  # (B, H*W, D)

        q = q.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, H*W, head_dim)
        k = k.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, H*W, head_dim)
        v = v.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, H*W, head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous().view(B, H * W, D)
        
        output = self.out(attn_output).permute(0, 2, 1).view(B, D, H, W)
        output = output * x1
        
        return output

class OCGNet(nn.Module):
    def __init__(self, emb_size=512, leaky=True):
        super(OCGNet, self).__init__()
        use_instnorm=False

        self.query_resnet = MyResnet()
        
        self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('./saved_models/yolov3.weights')
        self.combine_clickptns_conv1 = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)

        self.downsampling_pooling = nn.AvgPool2d(kernel_size=4, stride=4)
        self.combine_clickptns_conv2 = ConvBatchNormReLU(2, 1, 8, 8, 0, 1, leaky=leaky, instance=use_instnorm)
        self.queryreferencefusion = QueryReferenceFusion()
        self.crossview_fusionmodule = CrossViewFusionModule()


        self.query_visudim = 512 
        self.reference_visudim = 512
        self.query_mapping_visu = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm) 

        ## output head
        self.fcn_out = torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
                nn.Conv2d(emb_size//2, 9*5, kernel_size=1))

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        mat_clickptns = mat_clickptns.unsqueeze(1)
        query_imgs = self.combine_clickptns_conv1( torch.cat((query_imgs, mat_clickptns), dim=1) )
        mat_clickptns = self.downsampling_pooling(mat_clickptns)
        
        query_fvisu, early_features = self.query_resnet(query_imgs)
        reference_raw_fvisu = self.reference_darknet(reference_imgs)
        reference_fvisu = reference_raw_fvisu[1]
        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)
        early_features = torch.mean(early_features, dim=1, keepdim=True)

        # Qurey-positional enhanencement
        position_feature = self.combine_clickptns_conv2(torch.cat((early_features, mat_clickptns), dim=1))
        B, D, Hquery, Wquery = query_fvisu.shape
        B, D, Hreference, Wreference = reference_fvisu.shape
        reference_pooling = F.max_pool2d(reference_fvisu, kernel_size=8)

        # Query-reference enhanencement
        qr_fused_features = self.queryreferencefusion(query_fvisu, reference_pooling)

        # Late-stage embedding
        qr_fused_features = qr_fused_features * position_feature
        query_gvisu = torch.mean(qr_fused_features.view(B, D, Hquery*Wquery), dim=2, keepdims=False).view(B, D)
        fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
        attn_score = attn_score.squeeze(1)

        outbox = self.fcn_out(fused_features)

        return outbox, attn_score