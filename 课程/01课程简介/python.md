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

```{.python .input  n=6}
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydot

iris = load_iris()
test_idx = [0,50,100]
# 训练数据
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# 测试数据
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# 训练分类器
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_target)

from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         impurity=True) 
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].create_png()
graph[0].write_png("pruning_1.png")
```
