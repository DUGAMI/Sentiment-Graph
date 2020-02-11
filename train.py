import numpy as np
import torch

from torch import nn,optim
from torch_geometric.data import Data
from DataSet import HIN
from model import HGCN

hin=HIN()
graph=hin[0]


lossFunction=nn.BCELoss()
model=HGCN()
#model.load_state_dict(torch.load("model-(lr0.01)-epoch1000.pt"))

optimizer=optim.Adam(model.parameters(),lr=0.01)

for epoch in range(100):
    for graph in hin:
        x1, x2, edge_index, label = graph
        output=model(x1,x2,edge_index.t())

        optimizer.zero_grad()
        loss=lossFunction(output,label)
        loss.backward()

        print(float(loss),flush=True)
        optimizer.step()

torch.save(model.state_dict(),"model-(lr0.01)-epoch100-noPop.pt")




