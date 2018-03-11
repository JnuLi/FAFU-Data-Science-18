
# 数据重构


```python
import numpy as np
import pandas as pd
```

* stack


```python
df_obj = pd.DataFrame(np.random.randint(0,10, (5,2)), columns=['data1', 'data2'])
df_obj
```


```python
stacked = df_obj.stack()
print(stacked)
```


```python
print(type(stacked))
print(type(stacked.index))
```


```python
# 默认操作内层索引
stacked.unstack()
```


```python
# 通过level指定操作索引的级别
stacked.unstack(level=0)
```
