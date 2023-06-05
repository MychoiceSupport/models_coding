from layers import *
import torch
import math
# from geopy.distance import geodesic
# import xgboost as xgb
from time_util import *


class Guide_diff(nn.Module):
    def __init__(self, config, args, inputdim=2, seq_len=11, device='cpu'):
        super().__init__()
        self.time_dimension = args.dimension
        self.channels = args.channels
        self.device = device
        self.seq_len = seq_len
        self.num_layers = config['layers']
        self.output_dim = 2
        self.time_len = 24
        self.input_dim = inputdim
        self.final_step = 1
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection = Conv1d_with_init(self.channels, inputdim, 1)
        nn.init.zeros_(self.output_projection.weight)
        self.fcEmbedding = nn.Linear(self.input_dim, self.channels)
        self.time_linear = nn.Linear(self.seq_len * self.output_dim, self.output_dim)
        self.time_encoder = DeltaEncoder(dimension=args.dimension)
        self.start_encoder = SemanticEncoder(dimension=args.dimension, device=self.device)
        self.end_encoder = SemanticEncoder(dimension=args.dimension, device=self.device)
        self.layers = args.layers

    def batch_deal(self, batch):
        (_, _, delta_feature, time_feature) = batch
        shape0 = time_feature.shape[0]
        shape1 = time_feature.shape[1]
        delta_feature = self.time_encoder(delta_feature).to(self.device)
        time_feature_0 = self.start_encoder(time_feature[0].reshape(-1, )).reshape(shape0, shape1, -1)
        time_feature_1 = self.end_encoder(time_feature[1].reshape(-1, )).reshape(shape0, shape1, -1)
        new_time_feature = torch.cat([time_feature_0, time_feature_1], dim=-1)
        return delta_feature, new_time_feature

    def process_data(self, batch):
        (x, ground_truth, delta_feature, time_feature) = batch
        # ground_truth_data = ground_truth_data.cuda().to(self.device).float()
        # if extra_feature != None:
        #     extra_feature = extra_feature.cuda().to(self.device).float()
        # else:
        #     pass
        # delta_feature, time_feature = self.batch_deal(batch)
        x = torch.tensor(x.to(self.device), requires_grad=True, dtype=torch.float32).to(self.device)
        ground_truth = torch.tensor(ground_truth.to(self.device), dtype=torch.float32).to(self.device)
        delta_feature = torch.tensor(delta_feature.to(self.device), dtype=torch.float32).to(self.device)
        time_feature = torch.tensor(time_feature.to(self.device), dtype=torch.float32).to(self.device)
        return (x, ground_truth, delta_feature, time_feature)

    def forward(self, batch):
        raise Exception("You should implement the 类!")


class BiLSTM(Guide_diff):
    def __init__(self, config, inputdim=2, seq_len=11, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        super(BiLSTM, self).__init__(config, inputdim, seq_len, device)
        self.seq_rnn = nn.LSTM(input_size=self.channels + 2 * self.time_len,
                               hidden_size=self.channels + 2 * self.time_len, num_layers=config['layers'],
                               dropout=0.1, bidirectional=True, batch_first=True).cuda().to(self.device)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * self.channels * self.seq_len, 2 * self.channels * self.seq_len),
            nn.Dropout(0.1),
            nn.Linear(2 * self.channels * self.seq_len, self.output_dim),
        )

    def forward(self, batch):
        (x, _, time_delta, time_feature) = self.process_data(batch)
        # x = self.fcEmbedding(x)
        x = torch.cat([x, time_feature])
        # x = x.permute(0, 2, 1)  ##B,2,L
        x = x.cuda().to(self.device)
        print('x_in:', x.shape)
        output, _ = self.seq_rnn(x)

        # output,(_,_) = self.input_layer(x)
        # print(output.shape)
        # output = self.process_layer(output.permute(0,2,1))
        # print("这里比较重要:",output.shape)
        # print("查看相应的设备:",output.device)
        x = self.output_layer(output.reshape(output.shape[1], -1).to(self.device))
        return x


class TCN(Guide_diff):
    def __init__(self, config, args, inputdim=2, seq_len=11, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TCN, self).__init__(config, args, inputdim, seq_len, device)
        self.filter_conv = nn.ModuleList(
            [
                nn.Conv1d(self.channels, self.channels, 3, padding=1)
                for _ in range(self.layers)
            ]
        )
        self.gate_conv = nn.ModuleList(
            [
                nn.Conv1d(self.channels, self.channels, 3, padding=1)
                for _ in range(self.layers)
            ]
        )
        self.skip_convs = nn.ModuleList(
            [
                nn.Conv1d(self.channels, self.channels, 1)
                for _ in range(self.layers)
            ]
        )
        self.bm = nn.ModuleList(
            [nn.BatchNorm1d(self.channels) for _ in range(self.layers)]
        )

    def forward(self, batch):
        (x, groundTruth, time_delta, time_start_end) = self.process_data(batch)
        # time_delta, time_start_end = self.batch_deal(batch)
        # x = torch.cat([x, time_start_end], dim = -1)
        # x = x.reshape(B, L * self.channels)
        # x = self.fcEmbedding(x)
        x = x.permute(0, 2, 1)
        x = self.input_projection(x)
        skip = 0
        residual = x
        # print('x_in:', x.shape)
        for i in range(self.layers):
            filter = self.filter_conv[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_conv[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            x = x + residual
            x = self.bm[i](x)
        x = F.relu(skip)
        x = self.output_projection(x)  # (B,channel,L)
        x = self.time_linear(x.reshape(x.shape[0], -1))
        x = x.squeeze()
        return x


class GRU(Guide_diff):
    def __init__(self, config, args, inputdim=2, seq_len=11, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(GRU, self).__init__(config, args, inputdim, seq_len, device)
        self.seq_rnn = nn.GRU(input_size=self.input_dim + 2 * self.time_len, hidden_size=self.channels,
                              num_layers=self.layers,
                              bidirectional=False, batch_first=True).to(self.device)
        self.output_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear((self.channels) * self.seq_len, (self.channels) * self.seq_len),
            # nn.Dropout(0.1),
            nn.Linear((self.channels) * self.seq_len, self.output_dim),
        )

    def forward(self, batch):
        (x, _, time_delta, time_feature) = self.process_data(batch)
        # time_delta, time_start_end = self.batch_deal(batch)
        # x = x.permute(0, 2, 1)  ##B,2,L
        # print("查看特征:",x.shape, time_feature.shape)
        # x = self.fcEmbedding(x)
        x = torch.cat([x, time_feature], dim=-1)
        # print(x)
        x = x.to(self.device)
        x, _ = self.seq_rnn(x)
        x = self.output_layer(x.reshape(x.shape[0], -1).to(self.device))
        x = x.squeeze()
        return x


class FC(Guide_diff):
    def __init__(self, config, args, inputdim=2, seq_len=11, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(FC, self).__init__(config, args, inputdim, seq_len, device)

    def forward(self, batch):
        (x, groundTruth, time_delta, time_start_end) = self.process_data(batch)
        residual = x
        for i in range(len(self.residual_layers)):
            x = self.residual_layers[i](x)
            x = F.leaky_relu(x)
        x = x + residual
        x = self.output_projection(x)  # (B,channel,L)
        x = x.permute(0, 2, 1)
        return x


class Att(Guide_diff):
    def __init__(self, config, args, inputdim=2, seq_len=11, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Att, self).__init__(config, args, inputdim, seq_len, device)
        self.nheads = config["nheads"]
        self.residual_layers = nn.ModuleList(
            [
                AttBlock(
                    side_dim=self.channels,
                    channels=self.channels,
                    nheads=args.nheads,
                    device=args.device,
                )
                for _ in range(self.layers)
            ]
        )
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        # 新增的关于DPE的模型
        self.dpe_fc1 = nn.Linear(inputdim, self.channels)
        self.st_mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                    nn.Linear(128, self.nheads))
        self.dpe_fc2 = nn.Linear(self.channels, self.channels)

    def forward(self, batch, side_info=None):
        (x, groundTruth, time_delta, time_start_end) = self.process_data(batch)
        # time_delta = self.batch_deal(batch)
        # x = torch.cat([x,time_start_end], dim=-1)
        B, L, D = x.shape
        # side_info = torch.cat([time_delta, time_start_end],dim=-1)
        # 新增的DPE的计算
        x = x.permute(0, 2, 1)
        x = self.input_projection(x)
        if side_info != None:
            _, _, k, _ = side_info.shape
            h = self.nheads
            kernel = self.st_mlp(side_info)  # [B, L, k, nheads]
            kernel = kernel.reshape(-1, k, h)  # [B*L, k, nheads]
            kernel = kernel.permute(0, 2, 1).unsqueeze(2)  # [B*L, nheads, 1, k]
            data_x = self.dpe_fc1(x).transpose(1, 2).unsqueeze(-1)  # [B, channel, L, 1]
            tmp = F.unfold(data_x, (k, 1), 1, ((k - 1) // 2, 0), 1).reshape(B, self.channels, k,
                                                                            -1)  # [B, channel, k, L]
            tmp = tmp.permute(0, 3, 2, 1).reshape(-1, k, self.channels)  # [B*L, k, channel]
            v = self.dpe_fc2(tmp).reshape(B * L, k, h, self.channels // h).permute(0, 2, 1, 3)  # [B*L, h, k, d//h]
            side_info = (kernel @ v).squeeze().reshape(B, L, self.channels).permute(0, 2, 1)

        x = F.relu(x)  # [B, C, L]

        skip = []
        for i in range(len(self.residual_layers)):
            x, skip_connection = self.residual_layers[i](x, side_info)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.output_projection1(x)  # (B,channel,L)
        x = F.relu(x)
        x = self.output_projection(x)  # (B,channel,L)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.time_linear(x)
        return x
        # x = x.reshape(B, L * self.channels)


class AttBlock(nn.Module):
    def __init__(self, side_dim, channels, nheads, device=None):
        super().__init__()
        # self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.cond_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.forward_time = SequentialLearning(channels=channels, nheads=nheads)

    def forward(self, x, side_info=None):
        B, channel, L = x.shape
        base_shape = x.shape
        y = x

        y = self.forward_time(y, base_shape)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        if side_info != None:
            side_info = self.cond_projection(side_info)  # (B,2*channel,L)
            y = y + side_info

        # channel为2C的tensor一分为二，走一个gated activation
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip
