# 載入套件
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# 讀入資料集
iris = datasets.load_iris()
X = iris.data[:, 0:2] # sepal length(cm) 與 sepal width(cm)
y = iris.target

# 標準化
std_scaler = StandardScaler()
std_scaler.fit(X)
X_std = std_scaler.transform(X)

# 切分訓練測試資料
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.3, random_state = 0)

# 訓練感知器模型
ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)
ppn.fit(X_train, y_train)

# 準備標點的樣式與網格顏色
markers = ('x', 'o', 's')
colors = ('red', 'blue', 'green')
color_map = ListedColormap(colors)

# 定義 X 軸與 Y 軸的上下限
x1_min, x1_max = X_std[:, 0].min() - 1, X_std[:, 0].max() + 1
x2_min, x2_max = X_std[:, 1].min() - 1, X_std[:, 1].max() + 1

# 網格陣列
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))

# 繪圖
Z = ppn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha = 0.2, cmap = color_map)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
for i in np.unique(y):
    plt.scatter(x = X_std[y == i, 0], y = X_std[y == i, 1], marker = markers[i], alpha = 0.7, c = colors[i], label = i)
plt.xlabel('sepal length(standardized)')
plt.ylabel('sepal width(standardized)')
plt.legend(loc = 'upper left')
plt.show()