import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 读取CSV文件
data = pd.read_csv('meet.csv')
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_layer = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_scores = self.attention_layer(lstm_output)
        attention_scores = torch.tanh(attention_scores)
        attention_weights = self.softmax(attention_scores)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
# 提取时间特征
data['Time'] = pd.to_datetime(data['Date'])  # 假设时间列名为 'Time'
data['Month'] = data['Time'].dt.month
data['Day'] = data['Time'].dt.day


# 对时间特征进行周期编码
def encode_cyclic_features(df, column, max_value):
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df

data = encode_cyclic_features(data, 'Month', 12)
data = encode_cyclic_features(data, 'Day', 31)

# 提取'Value'列作为目标变量
values = data[['Price']].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# 创建时间步特征和跳跃选取'Value'的值
def create_time_features(data, n_steps_in, n_steps_out, n_jump):
    X, y = [], []
    for i in range(0, len(data) - n_steps_in - n_steps_out + 1, n_jump):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data):
            break
        seq_x = data[i:end_ix]
        seq_y = data[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# 设置时间步
n_steps_in = 30  # 输入序列长度
n_steps_out = 2  # 输出序列长度
n_jump = 2  # 跳跃步长

# 创建序列数据
X, y = create_time_features(scaled_values, n_steps_in, n_steps_out, n_jump)

# 划分训练集和测试集
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 添加时间编码特征
def add_time_features(X, time_data, n_steps_in):
    time_features = []
    for i in range(len(X)):
        start_idx = i * n_jump
        end_idx = start_idx + n_steps_in
        time_window = time_data[start_idx:end_idx]
        time_features.append(time_window)
    time_features = np.array(time_features)
    return np.concatenate((X, time_features), axis=2)

time_columns = ['Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']
time_data = data[time_columns].values

X_train = add_time_features(X_train, time_data, n_steps_in)
X_test = add_time_features(X_test, time_data, n_steps_in)
print("训练集的大小",X_train.shape)
# 定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze(-1)  # 确保 y 的形状是 (batch_size, output_size)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建数据加载器
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义多层LSTM-GRU混合模型
class HybridLSTMGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers, gru_layers, output_size):
        super(HybridLSTMGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.gru_layers = gru_layers
		#两层卷积层
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
		input_size = 64  # Adjust based on pooling layers
        #堆叠的LSTM层
        self.lstm_stack = nn.LSTM(input_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        # GRU层
        self.gru_stack = nn.GRU(hidden_size, hidden_size, num_layers=gru_layers, batch_first=True)
		#注意力层
        self.dropout = nn.Dropout(0.2)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化LSTM的隐藏状态和细胞状态
        h0_lstm = torch.zeros(self.lstm_layers, x.size(0), self.hidden_size).to(x.device)
        c0_lstm = torch.zeros(self.lstm_layers, x.size(0), self.hidden_size).to(x.device)
        # LSTM层
        lstm_out, _ = self.lstm_stack(x, (h0_lstm, c0_lstm))
        # 初始化GRU的隐藏状态
        h0_gru = torch.zeros(self.gru_layers, x.size(0), self.hidden_size).to(x.device)
        # GRU层
        gru_out, _ = self.gru_stack(lstm_out, h0_gru)
        # 添加Dropout
        gru_out = self.dropout(gru_out)
        # 注意力层
        context = self.attention(gru_out)  # 输出形状: (batch_size, hidden_size)
        # 全连接层
        out = self.fc(context)  # 输出形状: (batch_size, output_size)
        return out
# 设置超参数
input_size = X_train.shape[2]  # 输入特征数量（包括时间编码和stap变量）
hidden_size = 128  # 隐藏层大小
lstm_layers = 2  # LSTM层数
gru_layers = 3  # GRU层数
output_size = n_steps_out  # 输出序列长度

# 初始化模型
model = HybridLSTMGRUModel(input_size, hidden_size, lstm_layers, gru_layers, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练模型
num_epochs = 100  # 训练轮数
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        # 打印调试信息
        #print(f"Model output shape: {outputs.shape}, Target shape: {y_batch.shape}")
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    # 测试模型
    model.eval()
    with torch.no_grad():
        predictions = []
        true_values = []
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(y_batch.cpu().numpy())
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        # 反归一化
        predictions = predictions.reshape(-1, 1)
        true_values = true_values.reshape(-1, 1)
        predictions_actual = scaler.inverse_transform(predictions)
        true_values_actual = scaler.inverse_transform(true_values)
        # 计算误差
        mse = np.mean((true_values_actual - predictions_actual) ** 2)
        mae = np.mean(np.abs(true_values_actual - predictions_actual))
        print(f"均方误差 (MSE): {mse}")
        print(f"平均绝对误差 (MAE): {mae}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        # 绘制测试集的预测值和真实值
