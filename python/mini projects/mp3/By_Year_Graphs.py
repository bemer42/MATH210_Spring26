import numpy as np
import pandas as pd
from build_adj_mats import build_adj_mats
import pickle

Amm = build_adj_mats("MajMaj_Edges_by_Year.xlsx")
Amn = build_adj_mats("MajMin_Edges_by_Year.xlsx")
Ann = build_adj_mats("MinMin_Edges_by_Year.xlsx")



# SAVE
with open('A_by_year.pkl', 'wb') as f:
    pickle.dump((Amm, Amn, Ann), f)


