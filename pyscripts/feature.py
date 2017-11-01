# encoding=utf-8
# 测试
import numpy as np
import pandas as pd
import timeit

# 距离预测日期前 第1天、第2天、第3天、前3天、前5天、前7天、前10天、前14天、前21天、前28天、前40天、前50天、前59天 被购买次数
date_slots = np.array(
    [np.array([1]), np.array([2]), np.array([3]), np.arange(3) + 1, np.arange(5) + 1, np.arange(7) + 1,
     np.arange(10) + 1, np.arange(14) + 1, np.arange(21) + 1, np.arange(28) + 1, np.arange(40) + 1, np.arange(50) + 1,
     np.arange(59) + 1])
