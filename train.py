import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ExprGraphDataset
from model import TGTEncoder,TGTPretrain

GRAPH_DIR="dataset/pretrain/dgl_graphs"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=1000
LR=1e-5

def train():
    ds=ExprGraphDataset(GRAPH_DIR,mask_rate=0.15)
    dl=DataLoader(ds,batch_size=1,shuffle=True)

    g0=ds[0]
    num_genes=int(g0.nodes["gene"].data["token"].max().item())+1

    encoder=TGTEncoder(num_genes=num_genes).to(DEVICE)
    model=TGTPretrain(encoder).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)

    for ep in range(EPOCHS):
        model.train()
        tot=0.0
        for g in dl:
            g=g[0].to(DEVICE)
            pred=model(g)
            mask=g.nodes["gene"].data["mask"]
            target=g.nodes["gene"].data["target"]
            loss=F.mse_loss(pred[mask],target[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot+=loss.item()
        avg=tot/len(dl)
        print(f"epoch {ep+1}/{EPOCHS} loss {avg:.6f}")
        torch.save(encoder.state_dict(),f"encoder_epoch{ep+1}.pt")

if __name__=="__main__":
    train()
