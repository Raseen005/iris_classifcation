import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



data = pd.read_csv("iris.csv", index_col=0)

classes = data['Species'].unique()
class_weight = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y = data['Species']
)

class_weight = dict(zip(classes, class_weight))
#print("The Class Weight: ",class_weight)  /// The class weight is perfectly balanced


x = data[data.columns[:-1]]
y = data[data.columns[-1]]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LogisticRegression(
    max_iter=200,
    random_state=42
)

model.fit(x_train, y_train)

prediction = model.predict(x_test)



print('Classfication_Report', classification_report(y_test, prediction))

c_matrix = confusion_matrix(y_test, prediction)
classes = sorted(data.iloc[:, -1].unique()) 

plt.figure(figsize=(14,8))
sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=classes, yticklabels=classes)
plt.title("Report", fontweight='bold', fontsize=24)
plt.xlabel("Flowers", fontweight='bold', fontsize=14)
plt.ylabel("Flowers", fontweight='bold', fontsize=14)
plt.savefig("Final Report.png")
plt.show()