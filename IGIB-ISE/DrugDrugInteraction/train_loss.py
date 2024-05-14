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
from utils import get_stats, write_summary, write_summary_total
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 600,600
torch.set_num_threads(2)
df = pd.read_csv(f'data/raw_data/ZhangDDI_train.csv', sep=",")





from utils import create_batch_mask, get_roc_score


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
train_set = torch.load("./data/processed/{}_train.pt".format(args.dataset))

print("Dataset Loaded! ({:.4f} sec)".format(time.time() - start))
from models.CGIB_cont import CGIB
device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
model = CGIB(device = device, tau = args.tau, num_step_message_passing = args.message_passing,EM=args.EM_NUM).to(device)
PATH='state_dict_model0.0001,0.0001,{}.pth'.format(args.EM_NUM)
model.load_state_dict(torch.load(PATH))
train_loader = DataLoader(train_set, batch_size = 128, shuffle=False)
test_outputs, test_labels = [], []
i=0
for bc, samples in enumerate(train_loader):
    print(i*128)
    i=i+1
    masks = create_batch_mask(samples)
    output, _ = model([samples[0].to(device), samples[1].to(device), masks[0].to(device), masks[1].to(device)], test = True)
    
    test_outputs.append(output.reshape(-1).detach().cpu().numpy())
    test_labels.append(samples[2].reshape(-1).detach().cpu().numpy())

test_outputs = np.hstack(test_outputs)
test_labels = np.hstack(test_labels)

test_roc_score, test_ap_score, test_f1_score, test_acc_score = get_roc_score(test_outputs, test_labels)
print(test_acc_score)
