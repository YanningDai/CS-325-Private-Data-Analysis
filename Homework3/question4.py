import warnings
import os
import zipfile
import urllib
from pathlib import Path
from statistics import mean
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from opacus import PrivacyEngine
from opacus.layers import DPLSTM

import wandb
import pandas as pd

# 配置项
config = {
    "train_split": 0.8,
    "delta": 8e-5,
    "learning_rate": 2.0,
    "secure_mode": False,
    "embedding_size": 64,
    "hidden_size": 128,
    "num_lstm_layers": 1,
    "bidirectional_lstm": False,
    "VOCAB_SIZE": 256 + 3,  # 256 alternatives in one byte, plus 3 special characters.
    "NAMES_DATASET_URL": "https://download.pytorch.org/tutorial/data.zip",
    "DATA_DIR": "names"
}

# WandB sweep 配置
sweep_config = {
    "method": "grid",
    "parameters": {
        "epochs": {"values": [20, 40, 60]},
        "batch_size": {"values": [128, 256, 1024]},
        "clipping_threshold": {"values": [0.5, 1.0, 2.0]},
        # 其他固定参数
        "train_split": {"value": config["train_split"]},
        "delta": {"value": config["delta"]},
        "learning_rate": {"value": config["learning_rate"]},
        "secure_mode": {"value": config["secure_mode"]},
        "embedding_size": {"value": config["embedding_size"]},
        "hidden_size": {"value": config["hidden_size"]},
        "num_lstm_layers": {"value": config["num_lstm_layers"]},
        "bidirectional_lstm": {"value": config["bidirectional_lstm"]},
        "VOCAB_SIZE": {"value": config["VOCAB_SIZE"]},
        "NAMES_DATASET_URL": {"value": config["NAMES_DATASET_URL"]},
        "DATA_DIR": {"value": config["DATA_DIR"]}
    }
}

# 创建并运行 sweep
sweep_id = wandb.sweep(sweep_config, project="char-classifier-experiments")

# 全局变量，用于记录所有实验组合的结果
results = []

#====================================================================================================


class CharByteEncoder(nn.Module):
    """
    This encoder takes a UTF-8 string and encodes its bytes into a Tensor. It can also
    perform the opposite operation to check a result.
    Examples:
    >>> encoder = CharByteEncoder()
    >>> t = encoder('Ślusàrski')  # returns tensor([256, 197, 154, 108, 117, 115, 195, 160, 114, 115, 107, 105, 257])
    >>> encoder.decode(t)  # returns "<s>Ślusàrski</s>"
    """

    def __init__(self):
        super().__init__()
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.pad_token = "<pad>"

        self.start_idx = 256
        self.end_idx = 257
        self.pad_idx = 258

    def forward(self, s: str, pad_to=0) -> torch.LongTensor:
        """
        Encodes a string. It will append a start token <s> (id=self.start_idx) and an end token </s>
        (id=self.end_idx).
        Args:
            s: The string to encode.
            pad_to: If not zero, pad by appending self.pad_idx until string is of length `pad_to`.
                Defaults to 0.
        Returns:
            The encoded LongTensor of indices.
        """
        encoded = s.encode()
        n_pad = pad_to - len(encoded) if pad_to > len(encoded) else 0
        return torch.LongTensor(
            [self.start_idx]
            + [c for c in encoded]  # noqa
            + [self.end_idx]
            + [self.pad_idx for _ in range(n_pad)]
        )

    def decode(self, char_ids_tensor: torch.LongTensor) -> str:
        """
        The inverse of `forward`. Keeps the start, end, and pad indices.
        """
        char_ids = char_ids_tensor.cpu().detach().tolist()

        out = []
        buf = []
        for c in char_ids:
            if c < 256:
                buf.append(c)
            else:
                if buf:
                    out.append(bytes(buf).decode())
                    buf = []
                if c == self.start_idx:
                    out.append(self.start_token)
                elif c == self.end_idx:
                    out.append(self.end_token)
                elif c == self.pad_idx:
                    out.append(self.pad_token)

        if buf:  # in case some are left
            out.append(bytes(buf).decode())
        return "".join(out)

    def __len__(self):
        """
        The length of our encoder space. This is fixed to 256 (one byte) + 3 special chars
        (start, end, pad).
        Returns:
            259
        """
        return 259

class NamesDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)

        self.labels = list({langfile.stem for langfile in self.root.iterdir()})
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}
        self.encoder = CharByteEncoder()
        self.samples = self.construct_samples()

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

    def construct_samples(self):
        samples = []
        for langfile in self.root.iterdir():
            label_name = langfile.stem
            label_id = self.labels_dict[label_name]
            with open(langfile, "r") as fin:
                for row in fin:
                    samples.append(
                        (self.encoder(row.strip()), torch.tensor(label_id).long())
                    )
        return samples # 总数据集。
    # samples本身是一个列表，而不是张量。每一个 sample 是一个元组 (data, label)，其中：data是一个张量，label是一个张量

    def label_count(self):
        cnt = Counter()
        for _x, y in self.samples:
            label = self.labels[int(y)]
            cnt[label] += 1
        return cnt
    
class CharNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        output_size,
        num_lstm_layers=2,
        bidirectional=False,
        vocab_size=config["VOCAB_SIZE"],
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = DPLSTM(
            embedding_size,
            hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # -> [B, T, D]
        x, _ = self.lstm(x, hidden)  # -> [B, T, H]
        x = x[:, -1, :]  # -> [B, H]
        x = self.out_layer(x)  # -> [B, C]
        return x

def download_and_extract(dataset_url, data_dir):
    try:
        print("Downloading and extracting ...")
        filename = "data.zip"
        urllib.request.urlretrieve(dataset_url, filename)
        with zipfile.ZipFile(filename) as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(filename)
        print("Completed!")
    except Exception as e:
        print(f"Failed to download dataset: {e}")

def padded_collate(batch, padding_idx=0): # 一个batch的数据，是从samples里面来的，自身不是张量；最后返回的xy是张量。
    x = pad_sequence(
        [elem[0] for elem in batch], batch_first=True, padding_value=padding_idx
    )
    y = torch.stack([elem[1] for elem in batch]).long()

    return x, y

def train_and_log(model, criterion, optimizer, train_loader, epoch, privacy_engine=None, device="cuda:0"):
    accs, losses = [], []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        preds = logits.argmax(-1)
        n_correct = preds.eq(y).sum().item()
        accs.append(n_correct / len(y))
        losses.append(loss.item())

    avg_acc = mean(accs)
    avg_loss = mean(losses)

    # WandB logging
    log_data = {
        "Train/Accuracy": avg_acc,
        "Train/Loss": avg_loss,
        "Epoch": epoch
    }
    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(wandb.config.delta)
        log_data["Train/Epsilon"] = epsilon
    wandb.log(log_data)
    
    return avg_acc, avg_loss

def test_and_log(model, test_loader, privacy_engine=None, device="cuda:0"):
    accs = []
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(-1)
            n_correct = preds.eq(y).sum().item()
            accs.append(n_correct / len(y))
            losses.append(loss.item())

    avg_acc = mean(accs)
    avg_loss = mean(losses)

    # WandB logging
    log_data = {
        "Test/Accuracy": avg_acc,
        "Test/Loss": avg_loss
    }
    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(wandb.config.delta)
        log_data["Test/Epsilon"] = epsilon
    wandb.log(log_data)

    return avg_acc, avg_loss

def train_sweep():
    run_name = f"epochs={wandb.config.epochs}, batch_size={wandb.config.batch_size}, clipping_threshold={wandb.config.clipping_threshold}"
    with wandb.init(name=run_name) as run:
        config = wandb.config
        
        # 数据加载器
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, pin_memory=True, collate_fn=padded_collate)
        test_loader = DataLoader(test_ds, batch_size=2 * config.batch_size, pin_memory=True, collate_fn=padded_collate)

        # 带差分隐私的训练
        model_dp = CharNNClassifier(
            embedding_size=config.embedding_size, 
            hidden_size=config.hidden_size, 
            output_size=len(ds.labels),
            num_lstm_layers=config.num_lstm_layers,
            bidirectional=config.bidirectional_lstm,
            vocab_size=config.VOCAB_SIZE
        ).to(device)
        optimizer_dp = torch.optim.SGD(model_dp.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        privacy_engine = PrivacyEngine(secure_mode=config.secure_mode)
        model_dp, optimizer_dp, train_loader = privacy_engine.make_private_with_epsilon(
            module=model_dp,
            optimizer=optimizer_dp,
            data_loader=train_loader,
            max_grad_norm=config.clipping_threshold,
            target_delta=config.delta,
            target_epsilon=12.0,
            epochs=config.epochs
        )

        print(f"Training with privacy: epochs={config.epochs}, batch_size={config.batch_size}, clipping_threshold={config.clipping_threshold}")
        
        for epoch in range(config.epochs):
            train_and_log(model_dp, criterion, optimizer_dp, train_loader, epoch, privacy_engine, device=device)
            if epoch % 5 == 0:
                test_and_log(model_dp, test_loader, privacy_engine, device=device)

        # 记录带隐私保护训练的最终测试结果
        final_dp_test_acc, final_dp_test_loss = test_and_log(model_dp, test_loader, privacy_engine, device=device)

        results.append({
            "Epochs": config.epochs,
            "Batch Size": config.batch_size,
            "Clipping Threshold": config.clipping_threshold,
            "DP Test Accuracy": final_dp_test_acc,
            "DP Test Loss": final_dp_test_loss,
        })
        # 完成带差分隐私的 WandB 运行
        wandb.log({
            "Final DP Test Accuracy": final_dp_test_acc,
            "Final DP Test Loss": final_dp_test_loss
        })
        wandb.finish()   

def train_sweep_nodp():

    
    
    # 开始一个新的 WandB 运行，用于无差分隐私的训练
    with wandb.init() as run_nodp:
        
        config = wandb.config
        run_name = f"no privacy: epochs={config.epochs}, batch_size={config.batch_size}"
        run_nodp.name = run_name
        run_nodp.save()

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, pin_memory=True, collate_fn=padded_collate)
        test_loader = DataLoader(test_ds, batch_size=2 * config.batch_size, pin_memory=True, collate_fn=padded_collate)

        # 无差分隐私的训练
        model_nodp = CharNNClassifier(
            embedding_size=config.embedding_size, 
            hidden_size=config.hidden_size, 
            output_size=len(ds.labels),
            num_lstm_layers=config.num_lstm_layers,
            bidirectional=config.bidirectional_lstm,
            vocab_size=config.VOCAB_SIZE
        ).to(device)
        optimizer_nodp = torch.optim.SGD(model_nodp.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()
            
        print(f"Training without privacy: epochs={config.epochs}, batch_size={config.batch_size}")
        for epoch in range(config.epochs):
            train_and_log(model_nodp, criterion, optimizer_nodp, train_loader, epoch, privacy_engine=None, device=device)
            if epoch % 5 == 0:
                test_and_log(model_nodp, test_loader, privacy_engine=None, device=device)

            # 记录无隐私保护训练的最终测试结果
        final_nodp_test_acc, final_nodp_test_loss = test_and_log(model_nodp, test_loader, privacy_engine=None, device=device)

            # 将当前参数组合和结果添加到结果列表中
        results.append({
            "Epochs": config.epochs,
            "Batch Size": config.batch_size,
            "Clipping Threshold": config.clipping_threshold,
            "NoPrivacy Test Accuracy": final_nodp_test_acc,
            "NoPrivacy Test Loss": final_nodp_test_loss
        })

            # 结束无差分隐私的 WandB 运行
        wandb.log({
            "Final NoPrivacy Test Accuracy": final_nodp_test_acc,
            "Final NoPrivacy Test Loss": final_nodp_test_loss
        })
        wandb.finish()    
#====================================================================================================

# 下载并加载数据集
warnings.simplefilter("ignore")
download_and_extract(config["NAMES_DATASET_URL"], config["DATA_DIR"])
names_folder = os.path.join(config["DATA_DIR"], 'data', 'names')
ds = NamesDataset(names_folder)
train_len = int(config["train_split"] * len(ds))
test_len = len(ds) - train_len
train_ds, test_ds = torch.utils.data.random_split(ds, [train_len, test_len])

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 启动 sweep 以自动遍历所有组合
# wandb.agent(sweep_id, function=train_sweep)
wandb.agent(sweep_id, function=train_sweep_nodp)

# 打印所有组合的最终结果表格
results_df = pd.DataFrame(results)
print("\n所有实验组合的最终结果：")
print(results_df)

globals()