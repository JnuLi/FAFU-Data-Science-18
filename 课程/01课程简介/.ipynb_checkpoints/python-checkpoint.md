# Python包

## 决策树可视化

### 1、安装GraphViz
[GraphViz下载地址](https://share.weiyun.com/f9d71db14494178b12fddcbde3c9e349)

### 2、安装pyparsing和pydot包

```{.python .input}
# pip安装
pip install pyparsing pydot

# conda安装
conda install pyparsing pydot
```

### 3、演示代码

```{.python .input}
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree  
from sklearn.externals.six import StringIO  
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from io import StringIO
import pydot 
from IPython.display import display
from IPython.display import Image

df = pd.read_csv('Heart.csv', index_col=0)
predictors_df = df[['Age', 'RestBP','Chol' , 'Fbs', 'RestECG', 'MaxHR', 'ExAng']]
cat_predictors_df = df[['ChestPain','Thal',  'Sex']]
dummies_df = pd.get_dummies(cat_predictors_df)
dfpreds = predictors_df.join(dummies_df)

X = dfpreds
y_l = df.iloc[:,-1]
D = {"Yes":1, "No":0}
y = [1*D[y_] for y_ in y_l] 

idx = np.random.randint(len(X), size=len(X))
Xb = X.iloc[idx, :]
yb = np.array(y)[idx]

bg = DecisionTreeClassifier(max_depth=3)
bg.fit(Xb, yb)


dummy_io = StringIO() 
export_graphviz(bg, out_file = dummy_io, feature_names=X.columns,\
                class_names=['Yes', 'No'], proportion=True, filled=True)
(graph,)=pydot.graph_from_dot_data(dummy_io.getvalue())
display(Image(graph.create_png()))


graph.write_png("pruning_1.png")
```
