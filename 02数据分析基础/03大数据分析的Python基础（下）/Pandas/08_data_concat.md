
# 数据合并 concat


```python
import numpy as np
import pandas as pd
```

* NumPy的concat


```python
arr1 = np.random.randint(0, 10, (3, 4))
arr2 = np.random.randint(0, 10, (3, 4))

print(arr1)
print(arr2)
```


```python
np.concatenate([arr1, arr2])
```


```python
np.concatenate([arr1, arr2], axis=1)
```

* Series上的concat


```python
# index 没有重复的情况
ser_obj1 = pd.Series(np.random.randint(0, 10, 5), index=range(0,5))
ser_obj2 = pd.Series(np.random.randint(0, 10, 4), index=range(5,9))
ser_obj3 = pd.Series(np.random.randint(0, 10, 3), index=range(9,12))

print(ser_obj1)
print(ser_obj2)
print(ser_obj3)
```


```python
pd.concat([ser_obj1, ser_obj2, ser_obj3])
```


```python
pd.concat([ser_obj1, ser_obj2, ser_obj3], axis=1)
```


```python
# index 有重复的情况
ser_obj1 = pd.Series(np.random.randint(0, 10, 5), index=range(5))
ser_obj2 = pd.Series(np.random.randint(0, 10, 4), index=range(4))
ser_obj3 = pd.Series(np.random.randint(0, 10, 3), index=range(3))

print(ser_obj1)
print(ser_obj2)
print(ser_obj3)
```


```python
pd.concat([ser_obj1, ser_obj2, ser_obj3])
```


```python
pd.concat([ser_obj1, ser_obj2, ser_obj3], axis=1, join='inner')
```

* DataFrame上的concat


```python
df_obj1 = pd.DataFrame(np.random.randint(0, 10, (3, 2)), index=['a', 'b', 'c'],
                       columns=['A', 'B'])
df_obj2 = pd.DataFrame(np.random.randint(0, 10, (2, 2)), index=['a', 'b'],
                       columns=['C', 'D'])
print(df_obj1)
print(df_obj2)
```


```python
pd.concat([df_obj1, df_obj2])
```


```python
pd.concat([df_obj1, df_obj2], axis=1)
```
