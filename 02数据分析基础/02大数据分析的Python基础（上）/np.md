
# NumPy

* ndarray 多维数组


```python
import numpy as np

# 生成指定维度的随机多维数据
data = np.random.rand(2, 3)
print(data)
print(type(data))
```

* ndim, shape 和 dtype 属性


```python
print('维度个数', data.ndim)
print('各维度大小: ', data.shape)
print('数据类型: ', data.dtype)
```

* 创建ndarray


```python
# list转换为 ndarray
l = range(10)
data = np.array(l)
print(data)
print(data.shape)
print(data.ndim)
```


```python
# 嵌套序列转换为ndarray
l2 = [range(10), range(10)]
data = np.array(l2)
print(data)
print(data.shape)
```


```python
# np.zeros, np.ones 和 np.empty

# np.zeros
zeros_arr = np.zeros((3, 4))

# np.ones
ones_arr = np.ones((2, 3))

# np.empty
empty_arr = np.empty((3, 3))

# np.empty 指定数据类型
empty_int_arr = np.empty((3, 3), int)

print(zeros_arr)
print('-------------')
print(ones_arr)
print('-------------')
print(empty_arr)
print('-------------')
print(empty_int_arr)
```


```python
# np.arange()
print(np.arange(10))
```

* ndarray数据类型


```python
zeros_float_arr = np.zeros((3, 4), dtype=np.float64)
print(zeros_float_arr)
print(zeros_float_arr.dtype)

# astype转换数据类型
zeros_int_arr = zeros_float_arr.astype(np.int32)
print(zeros_int_arr)
print(zeros_int_arr.dtype)
```

* 矢量化 (vectorization)


```python
# 矢量与矢量运算
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print("元素相乘：")
print(arr * arr)

print("矩阵相加：")
print(arr + arr)
```


```python
# 矢量与标量运算
print(1. / arr)
print(2. * arr)
```

* 索引与切片


```python
# 一维数组
arr1 = np.arange(10)
print(arr1)

print(arr1[2:5])
```


```python
# 多维数组
arr2 = np.arange(12).reshape(3,4)
print(arr2)
```


```python
print(arr2[1])

print(arr2[0:2, 2:])

print(arr2[:, 1:3])
```


```python
# 条件索引

# 找出 data_arr 中 2015年后的数据
data_arr = np.random.rand(3,3)
print(data_arr)

year_arr = np.array([[2000, 2001, 2000],
                     [2005, 2002, 2009],
                     [2001, 2003, 2010]])

#is_year_after_2005 = year_arr >= 2005
#print is_year_after_2005, is_year_after_2005.dtype

#filtered_arr = data_arr[is_year_after_2005]

filtered_arr = data_arr[year_arr >= 2005]
print(filtered_arr)
```


```python
# 多个条件
filtered_arr = data_arr[(year_arr <= 2005) & (year_arr % 2 == 0)]
print(filtered_arr)
```

* 转置


```python
arr = np.random.rand(2,3)
print(arr)
print(arr.transpose())
```


```python
arr3d = np.random.rand(2,3,4)
print(arr3d)
print('----------------------')
print(arr3d.transpose((1,0,2))) # 3x2x4 
```

* 通用函数


```python
arr = np.random.randn(2,3)

print(arr)
print(np.ceil(arr))   #向上取整
print(np.floor(arr))   #向下取整
print(np.rint(arr))   #对四舍五入
print(np.isnan(arr))   #布尔
```

* np.where


```python
arr = np.random.randn(3,4)
print(arr)

np.where(arr > 0, 1, -1)
```

*  常用的统计方法


```python
arr = np.arange(10).reshape(5,2)
print(arr)

print(np.sum(arr))
print(np.sum(arr, axis=0))
print(np.sum(arr, axis=1))
```

* np.all 和 np.any


```python
arr = np.random.randn(2,3)
print(arr)

print(np.any(arr > 0))
print(np.all(arr > 0))
```

* np.unique


```python
arr = np.array([[1, 2, 1], [2, 3, 4]])
print(arr)
print(np.unique(arr))   #去重
```
