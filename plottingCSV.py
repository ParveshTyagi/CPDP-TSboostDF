import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('Results\Plotting Data\Scores - Balance.csv')

fig = plt.figure(figsize =(12, 7))

plt.boxplot(df, labels=df.columns)

plt.show()