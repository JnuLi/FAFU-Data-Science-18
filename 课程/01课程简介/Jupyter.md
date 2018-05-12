# Jupyter Notebook

## Jupyter插件

### notedown插件

运行Jupyter并加载notedown插件：


```bash
pip install https://github.com/mli/notedown/tarball/master
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

【可选项】默认开启notedown插件

首先生成jupyter配置文件（如果已经生成过可以跳过）


```bash
jupyter notebook --generate-config
```

将下面这一行加入到生成的配置文件的末尾（Linux/macOS一般在`~/.jupyter/jupyter_notebook_config.py`)


```bash
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

之后就只需要运行`jupyter notebook`即可。

### ExecutionTime插件

我们可以通过ExecutionTime插件来对每个cell的运行计时。

```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

### Jupyter 扩展包
[数据科学家效率提升必备技巧之Jupyter Notebook篇](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650737481&idx=2&sn=2b3f6df7c0c8b3d835c1a232ac4ad330&chksm=871acf37b06d4621d98d9fa88250b348f5e5cc6772764e4411cb85d32102ad303fbebdd043a6#rd)