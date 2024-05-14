import pandas as pd
import torch
from rdkit.Chem.Draw import SimilarityMaps
import random
import numpy as np
from utils import create_batch_mask
import os
import argument
import time
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from torch_geometric.data import DataLoader
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 600,600
torch.set_num_threads(2)
df = pd.read_csv(f'/home/fangjf/DISE_MOL/MolecularInteraction/data/raw_data/data_solve_data_Combisolv.csv', sep=",")








def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
args, unknown = argument.parse_args()
    
print("Loading dataset...")
start = time.time()

# Load dataset
train_set = torch.load("./data/processed/chr_abs.pt")

print("Dataset Loaded! ({:.4f} sec)".format(time.time() - start))
from models.CGIB_cont import CGIB
device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
model = CGIB(device = device, tau = args.tau, num_step_message_passing = args.message_passing,EM=args.EM_NUM).to(device)
PATH='state_dict_model{},{},20.pth'.format(0.000001,0.000001)
model.load_state_dict(torch.load(PATH))
train_loader = DataLoader(train_set, batch_size = 1, shuffle=False)



i=3822  


z =0  




for bc, samples in enumerate(train_loader):
    model.eval()

    masks = create_batch_mask(samples)
    
    solute_sublist,solvent_sublist = model.get_subgraph([samples[0].to(device), samples[1].to(device), masks[0].to(device), masks[1].to(device)],bottleneck=True)
    print(z)
    if z == i:
        print(df.iloc[i]["Solvent"])
        print(solvent_sublist)
        mol = Chem.MolFromSmiles(df.iloc[i]["Chromophore"])
        solu_atom_num=mol.GetNumAtoms()
        mol.RemoveAllConformers()
        for j in range(len(solute_sublist)):
            solvent_fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,solute_sublist[j].cpu().detach().numpy() ,colorMap='RdBu',
                                                            alpha=0.05,
                        size=(200,200))
            solvent_fig.savefig("solute/solute{}.png".format(j), bbox_inches='tight', dpi=600)


        mol = Chem.MolFromSmiles(df.iloc[i]["Solvent"])
        solu_atom_num=mol.GetNumAtoms()
        mol.RemoveAllConformers()
        for j in range(len(solvent_sublist)):
            solvent_fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,solvent_sublist[j].cpu().detach().numpy() ,colorMap='RdBu',
                                                            alpha=0.05,
                        size=(200,200))
            solvent_fig.savefig("solvent/solute{}.png".format(j), bbox_inches='tight', dpi=600)
    z=z+1
#print(1)
