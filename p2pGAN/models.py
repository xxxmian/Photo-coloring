import torch
import torch.nn as nn


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class D(nn.Module):
    def __init__(self, nc, nf):
        super(D, self).__init__()
        
        main = nn.Sequential()
        # 32
        layer_idx = 1
        name = 'layer%d' % layer_idx
        main.add_module('%s_conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))
        
        # 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        main.add_module(name, blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False))
        
        # 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module(name, blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False))
        
        # 4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, nf * 2, 4, 1, 1, bias=False))
        main.add_module('%s_bn' % name, nn.BatchNorm2d(nf * 2))
        
        # 3
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
        main.add_module('%s_sigmoid' % name, nn.Sigmoid())
        # 2 (sizePatchGAN=2)
        
        self.main = main
    
    def forward(self, x):
        output = self.main(x)
        return output


class G(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(G, self).__init__()
        
        # input is 32 x 32
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # input is 16 x 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8 x 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf * 2, nf * 4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf * 4, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 2
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        
        ## NOTE: decoder
        # input is 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8
        dlayer5 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=True)

        # import pdb; pdb.set_trace()
        # input is 2
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer4 = blockUNet(d_inc, nf * 4, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 4 * 2
        dlayer3 = blockUNet(d_inc, nf * 2, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 2 * 2
        dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        d_inc = nf * 2
        dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
        dlayer1.add_module('%s_tanh' % name, nn.Tanh())
        
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
    
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        
        dout5 = self.dlayer5(out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1
