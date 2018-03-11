
# Pandas层级索引


```python
import pandas as pd
import numpy as np
```


```python
ser_obj = pd.Series(np.random.randn(12),
                    index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd'],
                           [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]])
print(ser_obj)
```

* MultiIndex索引对象


```python
print(type(ser_obj.index))
print(ser_obj.index)
```

* 选取子集


```python
# 外层选取
print(ser_obj['c'])
```


```python
# 内层选取
print(ser_obj[:, 2])
```

* 交换分层顺序


```python
print(ser_obj.swaplevel())
```

* 交换并排序分层


```python
print(ser_obj.swaplevel().sortlevel())
```
