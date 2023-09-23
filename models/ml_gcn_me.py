import torchvision.models as models
from torch.nn import Parameter
from utils.util import *
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN_block(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super(GCN_block, self).__init__()
        self.gc1 = GraphConvolution(in_features, mid_features)
        self.gc2 = GraphConvolution(mid_features, out_features)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, inp, adj):
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        return x
#ref : https://arxiv.org/abs/2005.14480
#ref : https://github.com/jfhealthcare/Chexpert/blob/master/model/global_pool.py
def logsum_pool(feature, gamma):
    (N, C, H, W) = feature.shape

    m, _ = torch.max(feature, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)
    value0 = feature - m
    area = 1.0 / (H * W)
    g = gamma

    return m + 1 / g * torch.log(area * torch.sum(
            torch.exp(g * value0), dim=(-1, -2), keepdim=True))

class Attention(nn.Module):
    
    def __init__(self, n_hidden_enc=512, n_hidden_dec=512):
        
        super().__init__()
        
        self.h_hidden_enc = n_hidden_enc
        self.h_hidden_dec = n_hidden_dec
        
        self.W = nn.Linear(n_hidden_enc, n_hidden_dec, bias=False) 
        self.V = nn.Parameter(torch.rand(n_hidden_dec))
        
    
    def forward(self, expert, x):
        ''' 
            PARAMS:           
                hidden_dec:     [b, n_layers, n_hidden_dec]    (1st hidden_dec = encoder's last_h's last layer)                 
                last_layer_enc: [b, seq_len, n_hidden_enc * 2] 
            
            RETURN:
                att_weights:    [b, src_seq_len] 
        '''
        hidden_dec =  expert.unsqueeze(1)
        last_layer_enc = x.unsqueeze(1)
        #print('hidden_dec', hidden_dec.shape)
        #print('last_layer_enc', last_layer_enc.shape)
        batch_size = last_layer_enc.size(0)
        src_seq_len = last_layer_enc.size(1)

        hidden_dec = hidden_dec[:, -1, :].unsqueeze(1).repeat(1, src_seq_len, 1)         #[b, src_seq_len, n_hidden_dec]
        hidden_dec = hidden_dec.permute(1,2,0)
        last_layer_enc = last_layer_enc.permute(1,0,2)
        #print('hidden_dec',hidden_dec.shape)
        #print('last_layer_enc',last_layer_enc.shape)
        #print(torch.cat((hidden_dec, last_layer_enc), dim=1).shape)
        tanh_W_s_h = torch.tanh(self.W(torch.cat((hidden_dec, last_layer_enc), dim=1)))  #[b, src_seq_len, n_hidden_dec]
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)       #[b, n_hidde_dec, seq_len]
        #print('tanh_W_s_h',tanh_W_s_h.shape)
        V = self.V.repeat(batch_size, 1).unsqueeze(1).permute(1,0,2)  #[b, 1, n_hidden_dec]
        #print('V',V.shape)
        e = torch.bmm(V, tanh_W_s_h).squeeze(1)        #[b, seq_len]
        
        att_weights = F.softmax(e, dim=2)              #[b, src_seq_len]
        
        return att_weights


class GCNResnet(nn.Module):
    def __init__(self, model, block, num_classes, in_channel=300, t=0, adj_file=None, num_experts=4):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3
        )
        self.inplanes = 1024
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(32, 32) # 14 -> 16 because of input size change / 32 for 1024 1024

        # set multi expert
        self.num_experts = num_experts
        self.layer4s = nn.ModuleList([self._make_layer(block, 512, 3, stride=2) for _ in range(num_experts)])
        self.experts = nn.ModuleList([GCN_block(in_channel, 1024, 512) for _ in range(num_experts)])

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        
        #attention
        #self.attention = Attention(512, 512)
        self.attention = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder =nn.TransformerEncoder(self.attention, num_layers=1) 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes))

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind, inp, adj):
        x = (self.layer4s[ind])(x)
        #x = self.pooling(x)
        x = logsum_pool(x, 0.5)
        x = x.view(x.size(0), -1)
        self.feat.append(x)

        expert = (self.experts[ind])(inp, adj)
        expert = expert.transpose(0, 1)
        
        #x_ = torch.matmul(x, expert)
        
        x = x.unsqueeze(0)
        expert = expert.unsqueeze(0).permute(0,2,1)

        
        x = self.transformer_encoder(x)
        expert = self.transformer_encoder(expert)
        
                
        x = x.squeeze(0)
        expert = expert.squeeze(0).permute(1,0)

        
        x = torch.matmul(x, expert)
        #x+=x_
        return x

    def forward(self, feature, inp):
        feature = self.features(feature)

        # outputs of multi experts
        outs = []
        self.feat = []
        inp = inp[0]
        adj = gen_adj(self.A).detach()
        for ind in range(self.num_experts):
            outs.append(self._separate_part(feature, ind, inp, adj))\
            
        final_out = torch.stack(outs, dim=1).mean(dim=1)

        return {
                "output": final_out, 
                "feat": torch.stack(self.feat, dim=1),
                "logits": torch.stack(outs, dim=1)
            }

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.experts[0].parameters(), 'lr': lr},
                {'params': self.experts[1].parameters(), 'lr': lr}
                ]




def gcn_resnet101(num_classes, t, adj_file=None, in_channel=300, block=None):
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    return GCNResnet(model, block, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
