```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns',300)
pd.set_option('max_rows',500)
```

# Calculation of Basic Characteristic Variables
### Calculate WAP
$$ \text{wap} = \frac{\text{bid_price} * \text{ask_size} + \text{ask_price} * \text{bid_size}}{\text{bid_size} + \text{ask_size}} $$
### Calculate log return
$$ \text{log return}_{t_1,t_2} = LR_{t_1,t_2} = \log{\frac{S_{t_2}}{S_{t_1}}}  = \log{S_{t_2}} - \log{S_{t_1}}$$ 
### Calculate realized volatility
$$ \text{realized volatility} = \sigma = \sqrt{\sum_t{LR_{t-1,t}^2}} $$

