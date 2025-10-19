import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('iris.csv', index_col=0)

sns.set_theme(style='whitegrid')
fig, axs = plt.subplots(2,2, figsize=(14,8))
axs = axs.ravel()

for i in range(len(axs)):
    ax = axs[i]
    
    sns.scatterplot(data=data, x=data.index, y=data.iloc[:,i],hue=data.columns[-1], ax=ax, palette='viridis')
    ax.set_xlabel("Index")
    ax.set_ylabel(data.columns[i])
    
    ax.set_title(f'comparision between data')
    
    
    
plt.tight_layout()

plt.savefig("Individual Comparision.png")
plt.show()
sns.pairplot(data=data, hue=data.columns[-1], palette='viridis')
plt.suptitle("Comparision")
plt.savefig("Pairplot.png")
plt.show()


l_encoder = LabelEncoder()

data.iloc[:,-1] = l_encoder.fit_transform(data.iloc[:,-1])

plt.figure(figsize=(14,8))
correlation_matrix = data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',center=0)
plt.title("Correlation Between Feauters")
plt.savefig("Correlation Matrix.png")
plt.show()