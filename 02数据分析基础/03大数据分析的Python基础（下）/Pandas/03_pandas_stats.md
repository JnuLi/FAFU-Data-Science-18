
# Pandas统计计算和描述


```python
import numpy as np
import pandas as pd
```

* 常用的统计计算


```python
df_obj = pd.DataFrame(np.random.randn(5,4), columns = ['a', 'b', 'c', 'd'])
df_obj
```


```python
df_obj.sum()
```


```python
df_obj.max()
```


```python
df_obj.min(axis=1)
```

* 统计描述


```python
df_obj.describe()
```
