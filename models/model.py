import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, einsum
import os
from einops import rearrange
from models.convnext import ConvNeXt
from models.utils import load_ckpt, Interpolate, FeatureFusionBlock_custom


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score

class Aggregation(nn.Module):
    def __init__(self, head, in_token,max_pos_size = 100, dilate = [3,5,7], **kwargs):
        super().__init__()
        self.head = head
        self.dilate_layers = nn.ModuleList()
        dilate_outchannel = in_token//2
        for i in dilate:
            dilate_layer = nn.Conv2d(in_token, dilate_outchannel, kernel_size=3, stride=1, padding=i, dilation=i)
            self.dilate_layers.append(dilate_layer)

        self.to_qk =  nn.Conv2d(dilate_outchannel*3 + in_token, in_token * 2, 1, bias=False)
        self.scale = head ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pos_emb = RelPosEmb(max_pos_size, in_token//head)
        self.project = nn.Conv2d(in_token, in_token, 1, bias=False)
        self.to_v = nn.Conv2d(in_token, in_token, 1, bias=False)
        # gl-conv

    def forward(self, x, v = None):
        '''
        args: x: B, 768, 1/16H, 1/16W
        '''
        _, _, h, w = x.shape
        out = [x]
        for layer in self.dilate_layers:
            data = layer(x)
            out.append(data)
        out = torch.cat(out, 1)  
        qk= self.to_qk(out)

        q, k = qk.chunk(2, dim=1) #[B, L, C]        
        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=self.head), (q, k))

        q = self.scale * q
        sim_content =  einsum('b h x y d, b h u v d -> b h x y u v', q, k)
        sim_pos = self.pos_emb(q)
        sim = sim_content + sim_pos
        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)
        if v is None:
            v = self.to_v(x)
        else:
            v = self.to_v(v)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.head)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        out = self.project(out)
        out =  x + self.gamma * out

        return out


class CMANet(nn.Module):
    def __init__(self, is_train=True, encoder_scale = 'base',**kwargs):
        super().__init__()

        if encoder_scale == 'xlarge':
            self.depth= [3, 3, 27, 3]
            self.encoder_dims = [256, 512, 1024, 2048]
            ckpt_path = './weights/convnext_xlarge_22k_224.pth'
        elif encoder_scale == 'large':
            self.depth= [3, 3, 27, 3]
            self.encoder_dims = [192, 384, 768, 1536]
            ckpt_path = './weights/convnext_large_22k_224.pth'
        else:
            self.depth= [3, 3, 27, 3]
            self.encoder_dims = [128, 256, 512, 1024]
            ckpt_path = './weights/convnext_base_22k_224.pth'            
        self.encoder = ConvNeXt(depths=self.depth, dims=self.encoder_dims, **kwargs)
        if  is_train:
            if os.path.exists(ckpt_path):
                self.encoder = load_ckpt(self.encoder, ckpt_path, device='cuda' if torch.cuda.is_available() else 'cpu')
                print("load encoder premodel succuss!!!")
            else:
                print("Please download the pre-trained weights corresponding to ConvNeXt !!!")
        self.decoder = Decoder(heads_dim = [16,8,4], in_channels = [512, 256, 128], out_channels = 256)


    def forward(self, img):
        encoder_list =  self.encoder(img)
        out = self.decoder(encoder_list)
        return out.squeeze(1)


class Decoder(nn.Module):
    def __init__(self,heads_dim = [16,8,4], in_channels = [768, 384, 192], out_channels = 256, **kwargs):
        super().__init__()

        self.aggtion = Aggregation(heads_dim[0], in_channels[0]) #512

        self.msr = MultiScaleRefinement(heads_dim, in_channels, out_channels)

        features = out_channels
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    def forward(self, encoder_out):
        '''
        Args:
            en_out: 0: 1/4H W , 1: 1/8 HW, 1/16 HW
        out: 
            out: H W
        '''
        cost = self.aggtion(encoder_out[-1])
        up2 = self.msr(encoder_out, cost)

        out = self.head(up2)
        return out

class MultiScaleRefinement(nn.Module):
    def __init__(self,heads_dim, in_channels, out_channels, **kwargs):
        super().__init__()
        self.scale_16_delayer = nn.Conv2d(
                        in_channels[0],
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
        self.scale_16_refinenet  = _make_fusion_block(out_channels, use_bn = False)
        self.scale_16_translayer = nn.Conv2d(in_channels[0], out_channels, 1, bias=False)

        self.scale_8_refinenet  = _make_fusion_block(out_channels, use_bn = False)
        self.multiscale8_conv  = nn.Sequential(nn.Conv2d(in_channels[1] + in_channels[0]//4, out_channels, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(True))

        self.scale_4_refinenet = _make_fusion_block(out_channels, use_bn = False)
        self.multiscale4_conv   = nn.Sequential(nn.Conv2d(in_channels[2] + in_channels[0]//16, out_channels, kernel_size=3, stride=1, padding=1),
                                              nn.ReLU(True))

    def forward(self, encoder_out, cost):
        conv4, conv8, conv16 =  encoder_out

        cost16 = self.scale_16_translayer (cost)
        conv16 = self.scale_16_delayer(conv16)
        up8    = self.scale_8_refinenet (conv16, cost16)


        cost8 = F.pixel_shuffle(cost, 2)  # 64
        conv8 = self.multiscale8_conv(torch.cat([conv8, cost8],1))
        up4   = self.scale_8_refinenet (up8, conv8)

        cost4 = F.pixel_shuffle(cost, 4)  # 64
        conv4 = self.multiscale4_conv(torch.cat([conv4, cost4],1))
        up2   = self.scale_4_refinenet (up4, conv4)
        
        return up2



if __name__ == "__main__":
    model = CMANet(is_train=False)
    model.eval()
    from torch.autograd import Variable
    import time
    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    infre_time = []
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    with torch.no_grad():
        for i in range(100):
            left_ = Variable(torch.randn(1,3,640 , 640)).to(device)          
            print(i)
            start =time.clock()
            out = model(left_)
            in_time= time.clock()-start
            infre_time.append(in_time)
            print('Running time: %s Seconds'%(in_time))
    print(np.array(infre_time).mean())
