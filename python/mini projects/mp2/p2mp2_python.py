import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import pandas as pd

df = pd.read_excel("Omnitech_Data.xlsx", sheet_name=0)  # or sheet_name="Sheet1"

x = df["Degrees"].to_numpy()
y = df["Angular Velocity"].to_numpy()






# Export to the same file
# out = pd.DataFrame({"xsol": xsol, "ysol": ysol})

# with pd.ExcelWriter("Omnitech_Data.xlsx", mode="a", engine="openpyxl") as writer:
#     out.to_excel(writer, sheet_name="solutions", index=False)