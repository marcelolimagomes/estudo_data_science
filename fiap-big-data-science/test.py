import pandas as pd
import numpy as np
import matplotlib

l = []
max = 10000

for i in range(0, max):
    l.append(np.random.normal(size= 100))

df_s = pd.DataFrame(l)
df_s.hist(bins=100)
