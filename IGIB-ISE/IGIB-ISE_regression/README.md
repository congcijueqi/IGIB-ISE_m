# molecular interaction prediction Task

### Requirements

- Python version: 3.7.10
- Pytorch version: 1.8.1
- torch-geometric version: 1.7.0

**Absorption max (nm)**,  **Emission max (nm)**, and **Lifetime (ns)** column.
    - Make separate csv file for each column, and erase the NaN values for each column.
    - We log normalize the target value for **Lifetime (ns)** data due to its high skewness.
- For Solvation Free Energy datasets, create the dataset based on the **Source_all** column in the excel file.
    - Make separate csv file for each data source.
- Put each datasets into ``data/raw`` and run ``data.py`` file.
- Then, the python file will create ``{}.pt`` file in ``data/processed``.

### Hyperparameters
Following Options can be passed to `main.py`

`--dataset:`
Name of the dataset. Supported names are: chr_abs, chr_emi, chr_emi, mnsol, freesol, compsol, abraham, and combisolv.  
usage example :`--dataset chr_abs`

`--lr:`
Learning rate for training the model.  
usage example :`--lr 0.001`

`--epochs:`
Number of epochs for training the model.  
usage example :`--epochs 500`

`--beta_1:`
Hyperparameters for balance the trade-off between prediction and compression.  
usage example :`--beta_1 1.0`

`--beta_2:`
Hyperparameters for balance the trade-off between prediction and compression.  
usage example :`--beta_2 1.0`

`--tau:`
usage example :`--tau 1.0`

`--EM_NUM`
IN hyperparameter for $\text{ISE}$.  
usage example :`--EM_NUM 20`
