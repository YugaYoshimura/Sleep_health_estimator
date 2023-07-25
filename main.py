import pandas as pd
import torch
import pprint
pd.set_option('display.max_columns', 16)

def read_data():
    # データの読み込み
    sleep_csv = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", index_col=0, header=0)
    # 読み込んだデータの確認
    # print(sleep_csv.head())

    # NNで処理できるようにデータを変換
    sleep_data = sleep_csv.replace(
        {
            "Gender": {"Male": 0, "Female": 1},
            "BMI Category": {"Normal": 0, "Normal Weight": 0, "Overweight": 2, "Obese": 3},
            "Sleep Disorder": {None: 0, "Sleep Apnea": 1, "Insomnia": 2},
        }
    )

    sleep_data["Age"] = sleep_data["Age"] / 10
    sleep_data["Physical Activity Level"] = sleep_data["Physical Activity Level"] / 10
    sleep_data["Heart Rate"] = sleep_data["Heart Rate"]/10
    sleep_data["Daily Steps"] = sleep_data["Daily Steps"]/1000

    sleep_data = sleep_data.drop("Occupation", axis=1)

    # pprint.pprint(sleep_data)



    # 先週から足したとこ
    a = sleep_data["Blood Pressure"].str.partition("/")
    sleep_data["Blood Pressure High"] = a[0].astype(float)
    sleep_data["Blood Pressure Low"] = a[2].astype(float)

    sleep_data = sleep_data.drop("Blood Pressure", axis=1)

    sleep_data["Blood Pressure High"] = sleep_data["Blood Pressure High"] / 10
    sleep_data["Blood Pressure Low"] = sleep_data["Blood Pressure Low"] / 10

    pprint.pprint(sleep_data)


    # 変換後のデータの確認
    # pprint.pprint(sleep_data)

    return sleep_data


# データをPyTorchでの学習に利用できる形式に変換
def create_dataset_from_dataframe(sleep_data, target_tag="Quality of Sleep"):
    # "Quality of Sleep"の列を目的にする
    target = torch.tensor(sleep_data[target_tag].values, dtype=torch.float32).reshape(-1, 1)
    # "Quality of Sleep"以外の列を入力にする
    input = torch.tensor(sleep_data.drop(target_tag, axis=1).values, dtype=torch.float32)
    return input, target


# 4層順方向ニューラルネットワークモデルの定義
class FourLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h1 = torch.tanh(self.l1(x))
        h2 = torch.tanh(self.l2(h1))
        o = self.l3(h2)
        return o

def train_model(nn_model, input, target):
    # データセットの作成
    tips_dataset = torch.utils.data.TensorDataset(input, target)
    # バッチサイズ=25として学習用データローダを作成
    train_loader = torch.utils.data.DataLoader(tips_dataset, batch_size=25, shuffle=True)

    # オプティマイザ
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01, momentum=0.9)

    # データセット全体に対して10000回学習
    for epoch in range(10000):
        # バッチごとに学習する
        for x, y_hat in train_loader:
            y = nn_model(x)
            loss = torch.nn.functional.mse_loss(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 1000回に1回テストして誤差を表示
        if epoch % 1000 == 0:
            with torch.inference_mode():  # 推論モード（学習しない）
                y = nn_model(input)
                loss = torch.nn.functional.mse_loss(y, target)
                print(epoch, loss)



# データの準備
sleep_data = read_data()
input, target = create_dataset_from_dataframe(sleep_data)

# NNのオブジェクトを作成
nn_model = FourLayerNN(input.shape[1], 30, 1)
train_model(nn_model, input, target)

# 学習後のモデルの保存
# torch.save(nn_model.state_dict(), "nn_model.pth")

# 学習後のモデルのテスト
test_data = torch.tensor(
    [
        [
            1,  #  Gender "Male": 0, "Female": 1
            3.5,  # Age / 10
            6,  # Sleep Duration
            5.3,  # Physical Activity Level / 10
            6,  # Stress Level
            2,  # BMI Category {"Normal": 0, "Normal Weight": 0, "Overweight": 2, "Obese": 3}
            7.2, # Heart Rate / 10
            7.0, # Daily Steps /1000
            1, # Sleep Disorder {None: 0, "Sleep Apnea": 1, "Insomnia": 2}
            1.4, # Blood Pressure High
            8.0 # Blood Pressure Low
        ]
    ],
    dtype=torch.float32,
)
with torch.inference_mode():  # 推論モード（学習しない）
    print(nn_model(test_data))
