from torch.nn import Linear, ReLU, Sequential, Conv1d, MaxPool1d, Module, BatchNorm1d, Dropout


class CNNModel(Module):
    def __init__(self):
        super().__init__()

        self.cnnLayers = Sequential(
            # filter number-feature maps.:Geometric Mean sqrt(63*26) = 40 and it is better to have an power of 2 fo it can be 32 strid is 1 to cover all details
            # filter size : lets start with small size and try
            Conv1d(63, 32, 3, 1, 2),
            BatchNorm1d(32),
            ReLU(),

            Conv1d(32, 64, 3, 1, 2),
            BatchNorm1d(64),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            Conv1d(64, 128, 3, 1, 2),
            # same number of output channels (out_channels)(filter number-feature maps) as the previous layer's output channels.
            BatchNorm1d(128),
            ReLU(),
            Dropout(p=0.3),

            Conv1d(128, 256, 3, 1, 2),
            BatchNorm1d(256),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            Conv1d(256, 512, 5, 1, 2),
            BatchNorm1d(512),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            Conv1d(512, 512, 5, 1, 2),
            BatchNorm1d(512),
            ReLU(),
            Dropout(p=0.3),

        )

        self.linearLayers = Sequential(
            Linear(512, 26),  # not sure
            BatchNorm1d(26),
            ReLU(),
        )

    # Defining the forward pass
    def forward(self, x):
        # Pass through convolutional layers
        x = self.cnnLayers(x)

        # Flatten the output for the linear layers
        x = x.view(x.size(0), -1)

        # Pass through linear layers
        x = self.linearLayers(x)

        return x
