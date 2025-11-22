# ==========================
# Part 1 — Smoothen Stock Price
# ==========================
def part1():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    data = pd.read_csv("TSLA.csv")
    prices = data['Open'].values

    def smoothen_price(prices, window_size):
        smooth_prices = []
        for i in range(len(prices)):
            if i < window_size:
                smooth_prices.append(np.mean(prices[:i+1]))
            else:
                smooth_prices.append(np.mean(prices[i-window_size+1:i+1]))
        smooth_prices = np.array(smooth_prices)
        
        plt.figure(figsize=(12,6))
        plt.plot(prices, label="Original Price")
        plt.plot(smooth_prices, label=f"Smoothened Price (window={window_size})", linewidth=2)
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.title("Stock Price Smoothing")
        plt.legend()
        plt.savefig("smooth_stock_plot.png")
        plt.show()
        return smooth_prices

    smooth_prices = smoothen_price(prices, 100)


# ==========================
# Part 2 — Linear Regression
# ==========================
def part2():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    data = pd.read_csv("penguins.csv")
    data = data.dropna()

    X = data[['flipper_length_mm']]
    y = data['body_mass_g']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Part 2 — Linear Regression")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.4f}")

    plt.figure(figsize=(10,6))
    plt.scatter(X, y, color='blue', label='Actual Body Mass')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted Body Mass')
    plt.xlabel('Flipper Length (mm)')
    plt.ylabel('Body Mass (g)')
    plt.title('Linear Regression Plot')
    plt.legend()
    plt.savefig("Linear_Regression_plot.png")
    plt.show()


# ==========================
# Part 3 — Logistic Regression
# ==========================
def part3():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    data = pd.read_csv("diabetes.csv")

    # 特征和标签
    features = ['Glucose', 'BloodPressure', 'BMI']
    X = data[features]
    y = data['Outcome']

    # 划分训练集（前75%）和测试集（后25%）
    split_index = int(0.75 * len(data))
    feature_train = X[:split_index]
    outcome_train = y[:split_index]
    feature_test = X[split_index:]
    outcome_test = y[split_index:]

    # 训练模型
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(feature_train, outcome_train)

    # 预测
    outcome_pred = logreg.predict(feature_test)

    # 计算 False Positive / False Negative
    false_positive = np.sum((outcome_pred == 1) & (outcome_test == 0))
    false_negative = np.sum((outcome_pred == 0) & (outcome_test == 1))
    positive_correct = np.sum((outcome_pred == 1) & (outcome_test == 1)) / np.sum(outcome_test == 1) * 100

    print(f"Part 3 — Logistic Regression")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    print(f"Percentage of positive patients correctly predicted: {positive_correct:.2f}%")


# ==========================
# 主函数
# ==========================
if __name__ == "__main__":
    # 运行你想执行的部分
    part1()
    part2()
    part3()
