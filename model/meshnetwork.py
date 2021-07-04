import torch
import torch.nn as nn
import torch.nn.functional as F
from model.conv import ChebConv

from .inits import reset
from torch_scatter import scatter_add


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class Enblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, **kwargs):
        super(Enblock, self).__init__()
        self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, down_transform):
        out = F.elu(self.conv(x, edge_index))
        out = Pool(out, down_transform)
        return out


class Deblock(nn.Module):
    def __init__(self, in_channels, out_channels, K, **kwargs):
        super(Deblock, self).__init__()
        self.conv = ChebConv(in_channels, out_channels, K, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out, edge_index))
        return out


class AE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, edge_index,
                 down_transform, up_transform, K, **kwargs):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        # self.num_vert used in the last and the first layer of encoder and decoder
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    Enblock(in_channels, out_channels[idx], K, **kwargs))
            else:
                self.en_layers.append(
                    Enblock(out_channels[idx - 1], out_channels[idx], K,
                            **kwargs))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    Deblock(out_channels[-idx - 1], out_channels[-idx - 1], K,
                            **kwargs))
            else:
                self.de_layers.append(
                    Deblock(out_channels[-idx], out_channels[-idx - 1], K,
                            **kwargs))
        # reconstruction
        self.de_layers.append(
            ChebConv(out_channels[0], in_channels, K, **kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        # self.edge_index = self.edge_index.type_as(x)
        # self.down_transform = self.down_transform.type_as(x)
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.edge_index[i], self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def decoder(self, x):
        

        num_layers = len(self.de_layers)
        num_deblocks = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.edge_index[num_deblocks - i],
                          self.up_transform[num_deblocks - i])
            else:
                # last layer
                x = layer(x, self.edge_index[0])
        return x

    def forward(self, x):
        # x - batched feature matrix
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


class DisAE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, edge_index,
                 down_transform, up_transform, K, **kwargs):
        super(DisAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        # self.num_vert used in the last and the first layer of encoder and decoder
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    Enblock(in_channels, out_channels[idx], K, **kwargs))
            else:
                self.en_layers.append(
                    Enblock(out_channels[idx - 1], out_channels[idx], K,
                            **kwargs))
        # self.en_layers.append(
        #     nn.Linear(self.num_vert * out_channels[-1], latent_channels))
        self.idenc = nn.Sequential(nn.Linear(self.num_vert * out_channels[-1], latent_channels))
        self.expenc = nn.Sequential(nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels * 2, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    Deblock(out_channels[-idx - 1], out_channels[-idx - 1], K,
                            **kwargs))
            else:
                self.de_layers.append(
                    Deblock(out_channels[-idx], out_channels[-idx - 1], K,
                            **kwargs))
        # reconstruction
        self.de_layers.append(
            ChebConv(out_channels[0], in_channels, K, **kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        # self.edge_index = self.edge_index.type_as(x)
        # self.down_transform = self.down_transform.type_as(x)
        for i, layer in enumerate(self.en_layers):
            # if i != len(self.en_layers) - 1:
               
                x = layer(x, self.edge_index[i], self.down_transform[i])
            # else:
        
        x = x.view(x.shape[0], -1)
        expcode = self.expenc(x)
        idcode = self.idenc(x)
        return expcode, idcode

    def decoder(self, expcode, idcode):
        x = torch.cat([expcode, idcode], 1)
        num_layers = len(self.de_layers)
        num_deblocks = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.edge_index[num_deblocks - i],
                          self.up_transform[num_deblocks - i])
            else:
                # last layer
                x = layer(x, self.edge_index[0])
        return x

    def forward(self, A, B):
        # x - batched feature matrix
        Aexp, Aid = self.encoder(A)
        Bexp, Bid = self.encoder(B)
        outA = self.decoder(Aexp,Aid)
        outB = self.decoder(Bexp,Bid)
        outAB = self.decoder(Bexp,Aid)
        outBA = self.decoder(Aexp,Bid)
        return outA, outB, outAB, outBA, Aexp, Aid, Bexp, Bid


class DisAE2(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, edge_index,
                 down_transform, up_transform, K, **kwargs):
        super(DisAE2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.down_transform = down_transform
        self.up_transform = up_transform
        # self.num_vert used in the last and the first layer of encoder and decoder
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    Enblock(in_channels, out_channels[idx], K, **kwargs))
            else:
                self.en_layers.append(
                    Enblock(out_channels[idx - 1], out_channels[idx], K,
                            **kwargs))
        # self.en_layers.append(
        #     nn.Linear(self.num_vert * out_channels[-1], latent_channels))
        self.idenc = nn.Sequential(nn.Linear(self.num_vert * out_channels[-1], latent_channels))
        self.expenc = nn.Sequential(nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels , self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    Deblock(out_channels[-idx - 1], out_channels[-idx - 1], K,
                            **kwargs))
            else:
                self.de_layers.append(
                    Deblock(out_channels[-idx], out_channels[-idx - 1], K,
                            **kwargs))
        # reconstruction
        self.idde_layers.append(
            ChebConv(out_channels[0], in_channels, K, **kwargs))

        self.idde_layers = nn.ModuleList()
        self.idde_layers.append(
            nn.Linear(latent_channels , self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.idde_layers.append(
                    Deblock(out_channels[-idx - 1], out_channels[-idx - 1], K,
                            **kwargs))
            else:
                self.idde_layers.append(
                    Deblock(out_channels[-idx], out_channels[-idx - 1], K,
                            **kwargs))
        # reconstruction
        self.idde_layers.append(
            ChebConv(out_channels[0], in_channels, K, **kwargs))


        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        # self.edge_index = self.edge_index.type_as(x)
        # self.down_transform = self.down_transform.type_as(x)
        for i, layer in enumerate(self.en_layers):
            # if i != len(self.en_layers) - 1:
               
                x = layer(x, self.edge_index[i], self.down_transform[i])
            # else:
        
        x = x.view(x.shape[0], -1)
        expcode = self.expenc(x)
        idcode = self.idenc(x)
        return expcode, idcode

    def decoder(self, expcode, idcode):
        # x = torch.cat([expcode, idcode], 1)
        x = idcode
        num_layers = len(self.idde_layers)
        num_deblocks = num_layers - 2
        for i, layer in enumerate(self.idde_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.edge_index[num_deblocks - i],
                          self.up_transform[num_deblocks - i])
            else:
                # last layer
                x = layer(x, self.edge_index[0])
        exp = expcode
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                exp = layer(expcode)
                exp = exp.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                exp = layer(exp, self.edge_index[num_deblocks - i],
                          self.up_transform[num_deblocks - i])
            else:
                # last layer
                exp = layer(exp, self.edge_index[0])
        fianl = x + exp 
        return final, x

    def forward(self, A):
        # x - batched feature matrix
        Aexp, Aid = self.encoder(A)
        # Bexp, Bid = self.encoder(B)
        outA, idA = self.decoder(Aexp,Aid)
        # outB, idB = self.decoder(Bexp,Bid)
        # outAB = self.decoder(Bexp,Aid)
        # outBA = self.decoder(Aexp,Bid)
        return outA, idA # outB , outAB, outBA, Aexp, Aid, Bexp, Bid