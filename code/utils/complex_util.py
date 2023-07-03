from .complexPyTorch.complexLayers import (
    ComplexConv1d,
    ComplexBatchNorm1d,
    ComplexLinear,
    ComplexReLU,
    ComplexSigmoid,
    ComplexTanh,
    ComplexDropout,
)
import numpy as np
import torch
from torch import nn


dict_act_complex = {
    "sigmoid": ComplexSigmoid(),
    "relu": ComplexReLU(),
    "tanh": ComplexTanh(),
}


def complex_normalize(input, mean=None, std=None):
    """
    Perform complex normalization
    for test set norm, mean and std are given
    for training set norm, mean and std are given as None, so are calculated on the fly
    """
    real_value, imag_value = input.real, input.imag
    if mean is None or std is None:
        _mean = np.array(real_value.mean() + 1j * imag_value.mean()).astype(
            np.complex64
        )
        _std = np.array(real_value.std() + 1j * imag_value.std()).astype(np.complex64)
    else:
        _mean = mean
        _std = std
    real_norm = (real_value - _mean.real) / _std.real
    imag_norm = (imag_value - _mean.imag) / _std.imag

    return (real_norm + 1j * imag_norm).astype(np.complex64), _mean, _std


class NN_complex(nn.Module):
    def __init__(
        self, n_features, list_hidden, activation_fn, use_dropout=False, use_bn=False
    ):
        super().__init__()

        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.activation = activation_fn
        list_neurons = [n_features] + list_hidden

        layers = []
        # hidden layers
        for i in range(len(list_neurons) - 1):
            layers += self.hidden_layer_block(
                list_neurons[i],
                list_neurons[i + 1],
                final_layer=False,
                use_dropout=self.use_dropout,
                use_bn=self.use_bn,
            )
        # output layer
        layers += self.hidden_layer_block(list_neurons[-1], 1, final_layer=True)

        self.layers = nn.Sequential(*layers)

    def hidden_layer_block(
        self, input_dim, output_dim, final_layer=False, use_dropout=False, use_bn=False
    ):
        layers = []
        layers += [ComplexLinear(input_dim, output_dim)]
        if final_layer:
            return layers
        else:

            if use_bn:
                layers += [ComplexBatchNorm1d(output_dim)]
            if use_dropout:
                layers += [ComplexDropout(0.2)]
            # activation
            layers += [self.activation]
        return layers

    def forward(self, x):
        x = self.layers(x)
        return x.abs()  # return a real value


class NN_complex_concat(nn.Module):
    def __init__(
        self, n_features, list_hidden, activation_fn, use_dropout=False, use_bn=False
    ):
        super().__init__()

        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.activation = activation_fn
        list_neurons = [n_features] + list_hidden

        layers = []
        # hidden layers
        for i in range(len(list_neurons) - 1):
            layers += self.hidden_layer_block(
                list_neurons[i],
                list_neurons[i + 1],
                use_dropout=self.use_dropout,
                use_bn=self.use_bn,
            )

        self.layers = nn.Sequential(*layers)

        # output layer
        self.out = nn.Linear(list_neurons[-1] * 2, 1)

    def hidden_layer_block(
        self, input_dim, output_dim, use_dropout=False, use_bn=False
    ):
        layers = []
        layers += [ComplexLinear(input_dim, output_dim)]
        if use_bn:
            layers += [ComplexBatchNorm1d(output_dim)]
        if use_dropout:
            layers += [ComplexDropout(0.2)]
        # activation
        layers += [self.activation]
        return layers

    def forward(self, x):
        x = self.layers(x)
        # concat real and imag part of x and apply a linear layer to it to construct a real value
        x = torch.cat((x.real, x.imag), dim=1)
        pred = self.out(x)
        return pred  # return a real value


class NN_complex_fe(nn.Module):
    def __init__(
        self,
        n_features,
        list_hidden,
        activation_fn,
        device,
        use_dropout=False,
        use_bn=False,
    ):
        super().__init__()
        self.n_features = n_features
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.activation = activation_fn
        list_neurons = [n_features] + list_hidden
        self.device = device
        self.weight = nn.Parameter(torch.randn(n_features)[None, :])

        layers = []
        # hidden layers
        for i in range(len(list_neurons) - 1):
            layers += self.hidden_layer_block(
                list_neurons[i],
                list_neurons[i + 1],
                use_dropout=self.use_dropout,
                use_bn=self.use_bn,
            )
        # output layer
        layers += self.hidden_layer_block(list_neurons[-1], 1, final_layer=True)

        self.layers = nn.Sequential(*layers)

    def hidden_layer_block(
        self, input_dim, output_dim, final_layer=False, use_dropout=False, use_bn=False
    ):
        layers = []
        layers += [ComplexLinear(input_dim, output_dim)]
        if final_layer:
            return layers
        else:

            if use_bn:
                layers += [ComplexBatchNorm1d(output_dim)]
            if use_dropout:
                layers += [ComplexDropout(0.2)]
            # activation
            layers += [self.activation]
        return layers

    def frequency_encoding_complex(self, n_features):
        # freq range from 3e8 to 15e8 in real world
        arr_freq = np.arange(0, n_features) * 0.1 + 3
        # normalize to [0,1]
        arr_freq_norm = arr_freq / arr_freq[-1]
        arr_freq_norm = np.array(
            arr_freq_norm + 1j * arr_freq_norm, dtype=np.complex64
        )[np.newaxis, :]
        return torch.from_numpy(arr_freq_norm).to(self.device)

    def forward(self, x):
        arr_freq_norm = self.frequency_encoding_complex(self.n_features)
        x += self.weight * arr_freq_norm
        x = self.layers(x)
        return x.abs()  # return a real value


class NN_complex_fe_concat(nn.Module):
    def __init__(
        self,
        n_features,
        list_hidden,
        activation_fn,
        device,
        use_dropout=False,
        use_bn=False,
    ):
        super().__init__()
        self.n_features = n_features
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.activation = activation_fn
        list_neurons = [n_features] + list_hidden
        self.device = device
        self.weight = nn.Parameter(torch.randn(n_features)[None, :])

        layers = []
        # hidden layers
        for i in range(len(list_neurons) - 1):
            layers += self.hidden_layer_block(
                list_neurons[i],
                list_neurons[i + 1],
                use_dropout=self.use_dropout,
                use_bn=self.use_bn,
            )

        self.layers = nn.Sequential(*layers)

        # output layer
        self.out = nn.Linear(list_neurons[-1] * 2, 1)

    def hidden_layer_block(
        self, input_dim, output_dim, use_dropout=False, use_bn=False
    ):
        layers = []
        layers += [ComplexLinear(input_dim, output_dim)]
        if use_bn:
            layers += [ComplexBatchNorm1d(output_dim)]
        if use_dropout:
            layers += [ComplexDropout(0.2)]
        # activation
        layers += [self.activation]
        return layers

    def frequency_encoding_complex(self, n_features):
        # freq range from 3e8 to 15e8 in real world
        arr_freq = np.arange(0, n_features) * 0.1 + 3
        # normalize to [0,1]
        arr_freq_norm = arr_freq / arr_freq[-1]
        arr_freq_norm = np.array(
            arr_freq_norm + 1j * arr_freq_norm, dtype=np.complex64
        )[np.newaxis, :]
        return torch.from_numpy(arr_freq_norm).to(self.device)

    def forward(self, x):
        arr_freq_norm = self.frequency_encoding_complex(self.n_features)
        x += self.weight * arr_freq_norm
        x = self.layers(x)

        # concat real and imag part of x and apply a linear layer to it to construct a real value
        x = torch.cat((x.real, x.imag), dim=1)
        pred = self.out(x)
        return pred  # return a real value


class CNN_baseline_complex(nn.Module):
    def __init__(self, base_channels, activation_fn, use_dropout=False, use_bn=False):
        super().__init__()

        self.conv1 = ComplexConv1d(
            in_channels=1,
            out_channels=base_channels,
            kernel_size=11,
            stride=5,
            padding=0,
        )
        self.conv2 = ComplexConv1d(
            in_channels=base_channels,
            out_channels=2 * base_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv3 = ComplexConv1d(
            in_channels=2 * base_channels,
            out_channels=4 * base_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv4 = ComplexConv1d(
            in_channels=4 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv5 = ComplexConv1d(
            in_channels=8 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        # self.pool = ComplexMaxPool1d(2, 2)

        self.fc = nn.Linear(
            in_features=1 * 2 * 8 * base_channels, out_features=1
        )  # TODO, 1 here is derived by the model structure

        self.conv_block1 = self.generate_conv_block(
            self.conv1, activation_fn, use_dropout, use_bn
        )
        self.conv_block2 = self.generate_conv_block(
            self.conv2, activation_fn, use_dropout, use_bn
        )
        self.conv_block3 = self.generate_conv_block(
            self.conv3, activation_fn, use_dropout, use_bn
        )
        self.conv_block4 = self.generate_conv_block(
            self.conv4, activation_fn, use_dropout, use_bn
        )
        self.conv_block5 = self.generate_conv_block(
            self.conv5, activation_fn, use_dropout, use_bn
        )

    def generate_conv_block(self, conv, activation_fn, use_dropout, use_bn):
        """conv-bn-activation-dropout"""
        layers = []
        layers.append(conv)
        if use_bn:
            layers.append(ComplexBatchNorm1d(conv.out_channels))
        layers.append(activation_fn)
        if use_dropout:
            layers.append(ComplexDropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x.real, x.imag), dim=1)
        x = self.fc(x)
        return x


class CNN_baseline_complex_fe(nn.Module):
    def __init__(
        self,
        n_features,
        base_channels,
        activation_fn,
        device,
        use_fe=False,
        use_dropout=False,
        use_bn=False,
    ):
        super().__init__()

        self.conv1 = ComplexConv1d(
            in_channels=1,
            out_channels=base_channels,
            kernel_size=11,
            stride=5,
            padding=0,
        )
        self.conv2 = ComplexConv1d(
            in_channels=base_channels,
            out_channels=2 * base_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv3 = ComplexConv1d(
            in_channels=2 * base_channels,
            out_channels=4 * base_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv4 = ComplexConv1d(
            in_channels=4 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv5 = ComplexConv1d(
            in_channels=8 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        # self.pool = ComplexMaxPool1d(2, 2)

        self.fc = nn.Linear(
            in_features=1 * 2 * 8 * base_channels, out_features=1
        )  # TODO, 1 here is derived by the model structure

        self.conv_block1 = self.generate_conv_block(
            self.conv1, activation_fn, use_dropout, use_bn
        )
        self.conv_block2 = self.generate_conv_block(
            self.conv2, activation_fn, use_dropout, use_bn
        )
        self.conv_block3 = self.generate_conv_block(
            self.conv3, activation_fn, use_dropout, use_bn
        )
        self.conv_block4 = self.generate_conv_block(
            self.conv4, activation_fn, use_dropout, use_bn
        )
        self.conv_block5 = self.generate_conv_block(
            self.conv5, activation_fn, use_dropout, use_bn
        )

        # frequency encoding
        self.use_fe = use_fe
        self.n_features = n_features
        self.device = device
        self.weight = nn.Parameter(torch.randn(n_features)[None, :].unsqueeze(1))

    def generate_conv_block(self, conv, activation_fn, use_dropout, use_bn):
        """conv-bn-activation-dropout"""
        layers = []
        layers.append(conv)
        if use_bn:
            layers.append(ComplexBatchNorm1d(conv.out_channels))
        layers.append(activation_fn)
        if use_dropout:
            layers.append(ComplexDropout(0.5))
        return nn.Sequential(*layers)

    def frequency_encoding_complex(self, n_features):
        # freq range from 3e8 to 15e8 in real world
        arr_freq = np.arange(0, n_features) * 0.1 + 3
        # normalize to [0,1]
        arr_freq_norm = arr_freq / arr_freq[-1]
        arr_freq_norm = np.array(
            arr_freq_norm + 1j * arr_freq_norm, dtype=np.complex64
        )[np.newaxis, :]
        return torch.from_numpy(arr_freq_norm).to(self.device)

    def forward(self, x):
        # frequency encoding
        if self.use_fe:
            arr_freq_norm = self.frequency_encoding_complex(self.n_features)
            x += self.weight * arr_freq_norm
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x.real, x.imag), dim=1)
        x = self.fc(x)
        return x

# implementation of an equivalent model to CNN_complex_fe without complex layers
class CNN_count(nn.Module):
    def __init__(
        self,
        n_features, 
        base_channels,
        activation_fn, 
        device,
        use_fe=False,
        use_dropout=False, 
        use_bn=False
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels =1, out_channels=base_channels, kernel_size=11, stride=5, padding=0)
        self.conv1_ = nn.Conv1d(in_channels =1, out_channels=base_channels, kernel_size=11, stride=5, padding=0)
        self.conv2 = nn.Conv1d(in_channels =base_channels, out_channels=2*base_channels, kernel_size=3, stride=2, padding=0)
        self.conv2_ = nn.Conv1d(in_channels =base_channels, out_channels=2*base_channels, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv1d(in_channels =2*base_channels, out_channels=4*base_channels, kernel_size=3, stride=2, padding=0)
        self.conv3_ = nn.Conv1d(in_channels =2*base_channels, out_channels=4*base_channels, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv1d(in_channels =4*base_channels, out_channels=8*base_channels, kernel_size=3, stride=1, padding=0)
        self.conv4_ = nn.Conv1d(in_channels =4*base_channels, out_channels=8*base_channels, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv1d(in_channels =8*base_channels, out_channels=8*base_channels, kernel_size=3, stride=1, padding=0)
        self.conv5_ = nn.Conv1d(in_channels =8*base_channels, out_channels=8*base_channels, kernel_size=3, stride=1, padding=0)

        # self.pool = ComplexMaxPool1d(2, 2)

        self.fc = nn.Linear(in_features=1*2*8*base_channels, out_features=1) # TODO, 1 here is derived by the model structure

        self.conv_block1 = self.generate_conv_block(self.conv1, activation_fn, use_dropout, use_bn)
        self.conv_block1_ = self.generate_conv_block(self.conv1_, activation_fn, use_dropout, use_bn)
        self.conv_block2 = self.generate_conv_block(self.conv2, activation_fn, use_dropout, use_bn)
        self.conv_block2_ = self.generate_conv_block(self.conv2_, activation_fn, use_dropout, use_bn)
        self.conv_block3 = self.generate_conv_block(self.conv3, activation_fn, use_dropout, use_bn)
        self.conv_block3_ = self.generate_conv_block(self.conv3_, activation_fn, use_dropout, use_bn)
        self.conv_block4 = self.generate_conv_block(self.conv4, activation_fn, use_dropout, use_bn)
        self.conv_block4_ = self.generate_conv_block(self.conv4_, activation_fn, use_dropout, use_bn)
        self.conv_block5 = self.generate_conv_block(self.conv5, activation_fn, use_dropout, use_bn)
        self.conv_block5_ = self.generate_conv_block(self.conv5_, activation_fn, use_dropout, use_bn)

        # frequency encoding
        self.use_fe = use_fe
        self.n_features = n_features
        self.device = device
        self.weight = nn.Parameter(torch.randn(n_features)[None, :].unsqueeze(1))

    def generate_conv_block(self, conv,activation_fn, use_dropout, use_bn):
        """conv-bn-activation-dropout"""
        layers = []
        layers.append(conv) 
        if use_bn:
            layers.append(nn.BatchNorm1d(conv.out_channels))
        layers.append(activation_fn)
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def frequency_encoding(self, n_features):
        # freq range from 3e8 to 15e8 in real world
        arr_freq = np.arange(0, n_features // 2) * 0.1 + 3
        # normalize to [0,1]
        arr_freq_norm = arr_freq / arr_freq[-1]
        arr_freq_norm = arr_freq_norm[np.newaxis, :].repeat(2, axis=1)
        return torch.Tensor(arr_freq_norm).to(self.device)
    
    def apply_complex(self, fr, fi, x, x_):
        return fr(x) - fi(x_), fr(x_) + fi(x)

    def forward(self, x):
        # frequency encoding
        if self.use_fe:
            arr_freq_norm = self.frequency_encoding(self.n_features)
            x += self.weight * arr_freq_norm
        x, x_ = torch.split(x, 121, dim=-1)
        x, x_ = self.apply_complex(self.conv_block1, self.conv_block1_, x, x_)
        x, x_ = self.apply_complex(self.conv_block2, self.conv_block2_, x, x_)
        x, x_ = self.apply_complex(self.conv_block3, self.conv_block3_, x, x_)
        x, x_ = self.apply_complex(self.conv_block4, self.conv_block4_, x, x_)
        x, x_ = self.apply_complex(self.conv_block5, self.conv_block5_, x, x_)
        
        x = torch.flatten(torch.cat((x,x_),dim=-1),1)
        x = self.fc(x)
        return x