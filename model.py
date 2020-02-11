import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree

class HINConv(MessagePassing):
    def __init__(self,Hindex,in_channels,out_channels):
        super(HINConv,self).__init__(aggr='add')

        self.danmu=torch.nn.Linear(in_channels,out_channels)
        self.geng = torch.nn.Linear(in_channels, out_channels)
        self.Hindex=Hindex

    def forward(self,x1,x2,edge_index):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x1.size(0)+x2.size(0))

        danmu=self.danmu(x1)
        geng=self.geng(x2)

        common=torch.cat((danmu,geng))

        return self.propagate(edge_index,x=common)

    def message(self,x_j,edge_index,size):

        row,col=edge_index
        deg=degree(row,size[0],dtype=x_j.dtype)
        deg_inv_sqrt=deg.pow(-0.5)
        norm=deg_inv_sqrt[row]*deg_inv_sqrt[col]

        return norm.view(-1,1)*x_j

    def update(self,aggr_out):

        aggr_out=F.leaky_relu(aggr_out)
        return aggr_out.split(self.Hindex,dim=0)

# class HINConv2():
#     def __init__(self, Hindex):
#
#         self.danmu = torch.nn.Linear(300, 30)
#         self.geng = torch.nn.Linear(300, 30)
#         self.Hindex = Hindex
#
#     def forward(self, x1, x2, edge_index):
#
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x1.size(0) + x2.size(0))
#
#         danmu = self.danmu(x1)
#         geng = self.geng(x2)


#异构图神经网络

class HGCN(torch.nn.Module):
    def __init__(self):
        super(HGCN, self).__init__()

        self.conv1=HINConv(25453,200,30)
        self.conv2=HINConv(25453,30,7)

    def forward(self,x1,x2,edge_index):

        x1,x2=self.conv1(x1,x2,edge_index)
        x1,x2=self.conv2(x1,x2,edge_index)

        return F.softmax(x1)