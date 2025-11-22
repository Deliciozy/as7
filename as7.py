import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV
data = pd.read_csv("TSLA.csv")
print(data.head())  # 查看前几行，确认列名

# 提取开盘价列
prices = data['Open'].values  # 如果你的列名不是 'Open'，请改成实际列名

# 定义平滑函数
def smoothen_price(prices, window_size):
    smooth_prices = []
    for i in range(len(prices)):
        if i < window_size:
            smooth_prices.append(np.mean(prices[:i+1]))
        else:
            smooth_prices.append(np.mean(prices[i-window_size+1:i+1]))
    
    smooth_prices = np.array(smooth_prices)
    
    # 绘图并保存
    plt.figure(figsize=(12,6))
    plt.plot(prices, label="Original Price")
    plt.plot(smooth_prices, label=f"Smoothened Price (window={window_size})", linewidth=2)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Stock Price Smoothing")
    plt.legend()
    plt.savefig("smooth_stock_plot.png")  # 保存为图片
    plt.show()  # 如果终端能弹窗可以显示，否则只保存
    
    return smooth_prices

# 测试函数
smooth_prices = smoothen_price(prices, 100)
