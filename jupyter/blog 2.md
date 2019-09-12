Title: My Second Blog
Date: 2019-09-12
Slug: blog-2


```python
import pandas as pd
import chart
from IPython.core.display import HTML
```


```python
df = pd.DataFrame([
    ['Ryan', 'a'],
    ['Hock', 'b'],
    ['Kavitha', 'c'],
    ['Anika', 'd'],
    ['Benoit', 'e']
])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ryan</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hock</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kavitha</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anika</td>
      <td>d</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Benoit</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>




```python
HTML(df.to_html())
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ryan</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hock</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kavitha</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anika</td>
      <td>d</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Benoit</td>
      <td>e</td>
    </tr>
  </tbody>
</table>



![](https://vignette.wikia.nocookie.net/leagueoflegends/images/c/c4/Nunu_OriginalSkin.jpg/revision/latest?cb=20180814202011)
