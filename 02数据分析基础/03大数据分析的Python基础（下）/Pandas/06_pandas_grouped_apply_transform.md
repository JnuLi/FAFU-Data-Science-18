
# 数据分组运算


```python
import pandas as pd
import numpy as np
```


```python
# 分组运算后保持shape
dict_obj = {'key1' : ['a', 'b', 'a', 'b', 
                      'a', 'b', 'a', 'a'],
            'key2' : ['one', 'one', 'two', 'three',
                      'two', 'two', 'one', 'three'],
            'data1': np.random.randint(1, 10, 8),
            'data2': np.random.randint(1, 10, 8)}
df_obj = pd.DataFrame(dict_obj)
df_obj
```


```python
# 按key1分组后，计算data1，data2的统计信息并附加到原始表格中
k1_sum = df_obj.groupby('key1').sum().add_prefix('sum_')
k1_sum
```


```python
# 方法1，使用merge
pd.merge(df_obj, k1_sum, left_on='key1', right_index=True)
```

* transform方法


```python
# 方法2，使用transform
k1_sum_tf = df_obj.groupby('key1').transform(np.sum).add_prefix('sum_')
df_obj[k1_sum_tf.columns] = k1_sum_tf
df_obj
```


```python
# 自定义函数传入transform
def diff_mean(s):
    """
        返回数据与均值的差值
    """
    return s - s.mean()

df_obj.groupby('key1').transform(diff_mean)
```


```python
dataset_path = './starcraft.csv'
df_data = pd.read_csv(dataset_path, usecols=['LeagueIndex', 'Age', 'HoursPerWeek', 
                                             'TotalHours', 'APM'])
```

* apply


```python
def top_n(df, n=3, column='APM'):
    """
        返回每个分组按 column 的 top n 数据
    """
    return df.sort_values(by=column, ascending=False)[:n]

df_data.groupby('LeagueIndex').apply(top_n)
```


```python
# apply函数接收的参数会传入自定义的函数中
df_data.groupby('LeagueIndex').apply(top_n, n=2, column='Age')
```

* 禁止分组 group_keys=False


```python
df_data.groupby('LeagueIndex', group_keys=False).apply(top_n)
```
