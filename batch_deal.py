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
        self.time_encoder = DeltaEncoder(dimension=args.dimension, device=self.device)
        self.start_encoder = SemanticEncoder(dimension=args.dimension, device=self.device)
        self.end_encoder = SemanticEncoder(dimension=args.dimension, device=self.device)
        self.layers = args.layers
        self.use_time = args.use_time

    def batch_deal(self, batch):
        (_, _, delta_feature, time_feature) = batch
        shape0 = time_feature.shape[0]
        shape1 = time_feature.shape[1]
        delta_feature = self.time_encoder(delta_feature).to(self.device)
        time_feature = time_feature.reshape(-1,2).cpu().numpy()
        start_time = pd.to_datetime(time_feature[:,0], unit='s')
        end_time = pd.to_datetime(time_feature[:,1], unit='s')
        week, day, month, hour = start_time.weekday.values.reshape(shape0, shape1), start_time.day.values.reshape(shape0, shape1), \
                                 start_time.month.values.reshape(shape0, shape1), start_time.hour.values.reshape(shape0, shape1)
        start_feature = self.start_encoder(week,day,month,hour).to(self.device)
        week, day, month, hour = end_time.weekday.values.reshape(shape0, shape1), end_time.day.values.reshape(shape0, shape1), \
                                 end_time.month.values.reshape(shape0, shape1), end_time.hour.values.reshape(shape0, shape1)
        end_feature = self.end_encoder(week,day,month,hour).to(self.device)
        # time_feature_0 = self.start_encoder(time_feature[0].reshape(-1, )).reshape(shape0, shape1, -1).to(self.device)
        # time_feature_1 = self.end_encoder(time_feature[1].reshape(-1, )).reshape(shape0, shape1, -1).to(self.device)
        new_time_feature = torch.cat([start_feature, end_feature], dim=-1).to(self.device)
        return delta_feature, new_time_feature

    def process_data(self, batch):
        (x, ground_truth, delta_feature, time_feature) = batch
        delta_feature, time_feature = self.batch_deal(batch)
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


class TCN(Guide_diff):
    def __init__(self, config,args,inputdim=2, seq_len=11, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TCN, self).__init__(config,args,inputdim, seq_len, device)
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
        (x, _, time_delta, time_feature) = self.process_data(batch)
        if self.use_time == True:
            time_delta, time_feature= self.batch_deal(batch)
            x = torch.cat([x, time_feature], dim = -1)
        # x = x.reshape(B, L * self.channels)
        # x = self.fcEmbedding(x)
        x = x.permute(0, 2, 1)
        x = self.input_projection(x)
        skip = 0
        residual = x
        #print('x_in:', x.shape)
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
        x = self.time_linear(x.reshape(x.shape[0],-1))
        x = x.squeeze()
        return x