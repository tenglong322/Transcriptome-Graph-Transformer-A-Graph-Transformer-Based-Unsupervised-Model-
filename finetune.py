import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import TGTEncoder,TGTFinetune
from finetune_dataset import GraphClsDataset

GRAPH_DIR="dataset/finetune/task1"
LABEL_CSV="dataset/finetune/task1.csv"
ENCODER_CKPT="encoder_epoch1000.pt"

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=30
LR=5e-4
BATCH_SIZE=1

def train():
    ds=GraphClsDataset(GRAPH_DIR,LABEL_CSV)
    dl=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)

    g0,_=ds[0]
    num_genes=int(g0.nodes["gene"].data["token"].max().item())+1
    num_classes=len(set(ds.label_map.values()))

    encoder=TGTEncoder(num_genes=num_genes).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_CKPT,map_location=DEVICE))

    model=TGTFinetune(encoder,num_classes).to(DEVICE)

    opt=torch.optim.AdamW(model.parameters(),lr=LR)

    for ep in range(EPOCHS):
        model.train()
        tot=0.0
        for g,y in dl:
            g=g.to(DEVICE)
            y=torch.tensor([y],device=DEVICE)
            logit=model(g)
            loss=F.cross_entropy(logit,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot+=loss.item()

        print(f"epoch {ep+1}/{EPOCHS} loss {tot/len(dl):.6f}")
        torch.save(model.state_dict(),f"finetune_epoch{ep+1}.pt")

if __name__=="__main__":
    train()
