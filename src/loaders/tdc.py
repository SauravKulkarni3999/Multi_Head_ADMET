from tdc.single_pred import ADME
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list

def load_bbbp():
    """Loads the BBBP Dataset from TDC."""
    data = ADME(name = "BBBP_Martins")
    df = data.get_data()
    df = df.rename(columns={'Drug': 'smiles', 'Y': 'p_np'})
    return df

def load_herg():
    """Loads the HERG Dataset from TDC.""" 
    label_list = retrieve_label_name_list("herg_central")
    data = Tox(name = "HERG_Central")
    df = data.get_data()
    df = df.rename(columns={'Drug': 'smiles', 'Y': 'hERG_inhibition'})
    return df

def load_cyp3a4():
    """Loads the CYP3A4 Dataset from TDC."""
    data = ADME(name = "CYP3A4_Substrate_CarbonMangels")
    df = data.get_data()
    df = df.rename(columns={'Drug': 'smiles', 'Y': 'CYP3A4_inhibition'})
    return df

def load_freesolv():
    """Loads the FreeSolv Dataset from TDC."""
    data = ADME(name = "HydrationFreeEnergy_FreeSolv")
    df = data.get_data()
    df = df.rename(columns={'Drug': 'smiles', 'Y': 'hydration_free_energy'})
    return df
