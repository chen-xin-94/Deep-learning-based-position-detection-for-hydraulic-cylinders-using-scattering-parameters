import os
import pickle
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm  # avoid multiple bars
from prettytable import PrettyTable

dict_act = {
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(0.2),
    "tanh": nn.Tanh(),
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_table(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class NN(nn.Module):
    def __init__(
        self,
        n_features,
        list_hidden,
        activation_fn,
        use_dropout=False,
        use_bn=False,
        ues_ln=False,
    ):
        super().__init__()

        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.use_ln = ues_ln
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
                use_ln=self.use_ln,
            )
        # output layer
        layers += self.hidden_layer_block(list_neurons[-1], 1, final_layer=True)

        self.layers = nn.Sequential(*layers)

    def hidden_layer_block(
        self,
        input_dim,
        output_dim,
        final_layer=False,
        use_dropout=False,
        use_bn=False,
        use_ln=False,
    ):
        layers = []
        layers += [nn.Linear(input_dim, output_dim)]
        if final_layer:
            return layers
        else:

            if use_bn:
                layers += [nn.BatchNorm1d(output_dim)]
            if use_ln:
                layers += [nn.LayerNorm(output_dim)]
            if use_dropout:
                layers += [nn.Dropout(0.2)]
            # activation
            layers += [self.activation]
        return layers

    def forward(self, x):
        return self.layers(x)


class NN_fe(nn.Module):
    def __init__(
        self,
        n_features,
        list_hidden,
        activation_fn,
        device,
        use_dropout=False,
        use_bn=False,
        ues_ln=False,
    ):
        super().__init__()
        self.n_features = n_features
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.use_ln = ues_ln
        self.activation = activation_fn
        list_neurons = [n_features] + list_hidden
        self.device = device

        self.weight = nn.Parameter(  # You use nn.Parameter so that these weights can be optimized
            # Initiate the weights for the channels from a random normal distribution
            torch.randn(n_features)[None, :]  #
        )

        layers = []
        # hidden layers
        for i in range(len(list_neurons) - 1):
            layers += self.hidden_layer_block(
                list_neurons[i],
                list_neurons[i + 1],
                final_layer=False,
                use_dropout=self.use_dropout,
                use_bn=self.use_bn,
                use_ln=self.use_ln,
            )
        # output layer
        layers += self.hidden_layer_block(list_neurons[-1], 1, final_layer=True)

        self.layers = nn.Sequential(*layers)

    def hidden_layer_block(
        self,
        input_dim,
        output_dim,
        final_layer=False,
        use_dropout=False,
        use_bn=False,
        use_ln=False,
    ):
        layers = []
        layers += [nn.Linear(input_dim, output_dim)]
        if final_layer:
            return layers
        else:
            if use_bn:
                layers += [nn.BatchNorm1d(output_dim)]
            if use_ln:
                layers += [nn.LayerNorm(output_dim)]
            if use_dropout:
                layers += [nn.Dropout(0.2)]
            # activation
            layers += [self.activation]
        return layers

    def frequency_encoding(self, n_features):
        # freq range from 3e8 to 15e8 in real world
        arr_freq = np.arange(0, n_features // 2) * 0.1 + 3
        # normalize to [0,1]
        arr_freq_norm = arr_freq / arr_freq[-1]
        arr_freq_norm = arr_freq_norm[np.newaxis, :].repeat(2, axis=1)
        return torch.Tensor(arr_freq_norm).to(self.device)

    def forward(self, x):

        arr_freq_norm = self.frequency_encoding(self.n_features)
        # positional encoding
        x += self.weight * arr_freq_norm

        return self.layers(x)


class CNN_baseline(nn.Module):
    def __init__(self, base_channels, activation_fn, use_dropout=False, use_bn=False):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=base_channels,
            kernel_size=22,
            stride=11,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            in_channels=base_channels,
            out_channels=2 * base_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv3 = nn.Conv1d(
            in_channels=2 * base_channels,
            out_channels=4 * base_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv4 = nn.Conv1d(
            in_channels=4 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        # self.pool = nn.MaxPool1d(2, 2)

        self.fc = nn.Linear(in_features=128, out_features=1)  # TODO

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

    def generate_conv_block(self, conv, activation_fn, use_dropout, use_bn):
        """conv-bn-activation-dropout"""
        layers = []
        layers.append(conv)
        if use_bn:
            layers.append(nn.BatchNorm1d(conv.out_channels))
        layers.append(activation_fn)
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CNN_baseline_fe(nn.Module):
    def __init__(
        self,
        n_features,
        base_channels,
        device,
        activation_fn,
        use_fe=False,
        use_dropout=False,
        use_bn=False,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=base_channels,
            kernel_size=22,
            stride=11,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            in_channels=base_channels,
            out_channels=2 * base_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv3 = nn.Conv1d(
            in_channels=2 * base_channels,
            out_channels=4 * base_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.conv4 = nn.Conv1d(
            in_channels=4 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        # self.pool = nn.MaxPool1d(2, 2)

        self.fc = nn.Linear(in_features=128, out_features=1)  # TODO

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

        # frequency encoding
        self.use_fe = use_fe
        self.n_features = n_features
        self.device = device
        self.weight = nn.Parameter(torch.randn(n_features)[None, :])

    def generate_conv_block(self, conv, activation_fn, use_dropout, use_bn):
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

    def forward(self, x):
        # frequency encoding
        if self.use_fe:
            arr_freq_norm = self.frequency_encoding(self.n_features)
            x += self.weight * arr_freq_norm
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_NN(
    model,
    config,
    loss_fn,
    optimizer,
    scheduler,
    train_dataloader,
    test_dataloader,
    TEST_dataloader,
    device,
    save_folder,
    model_name,
    writer=None, # tensorboard SummaryWriter
    display_step=10,
    save = True,
):

    best_loss = 100000
    best_epoch = 0
    counter_early_stopping = 0

    train_losses = []  # for all config.epochs
    test_losses = []
    TEST_losses = []

    model_save_path = os.path.join(save_folder, model_name)
    for epoch in tqdm(range(config.epochs)):
        ## training
        model.train()

        _train_losses = []  # within in one epoch
        _test_losses = []
        _TEST_losses = []
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            ## Model computations
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _train_losses.append(loss.item() * inputs.size(0))  # append total loss
        train_losses.append(
            np.sqrt(np.sum(_train_losses) / len(train_dataloader.dataset))  # log RMSE
        )
        ## validation
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                _test_losses.append(loss.item() * inputs.size(0))  # append total loss
            test_losses.append(
                np.sqrt(np.sum(_test_losses) / len(test_dataloader.dataset))
            )  # log RMSE

        # evaluation on TEST sets
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(TEST_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                _TEST_losses.append(loss.item() * inputs.size(0))  # append total loss
            TEST_losses.append(
                np.sqrt(np.sum(_TEST_losses) / len(TEST_dataloader.dataset))
            )  # log RMSE

        ## tensorboard
        if writer:
            writer.add_scalar("Loss/NN/train", train_losses[epoch], epoch)
            writer.add_scalar("Loss/NN/test", test_losses[epoch], epoch)
            writer.add_scalar("Loss/NN/TEST", TEST_losses[epoch], epoch)

        ## save the model if it is the best
        is_best = test_losses[epoch] < best_loss
        if is_best:
            best_loss = test_losses[epoch]
            best_epoch = epoch
            counter_early_stopping = 0
            if save:
                torch.save(model.state_dict(), model_save_path)

        ## early stopping
        else:
            counter_early_stopping += 1
            if counter_early_stopping > config.patience:
                print("Early stopping triggered")
                break

        if epoch % display_step == 0:
            print(
                "epoch : {}, train loss : {:.2f}, test loss : {:.2f}, TEST loss : {:.2f}, lr: {:.4f}".format(
                    epoch,
                    train_losses[epoch],
                    test_losses[epoch],
                    TEST_losses[epoch],
                    optimizer.param_groups[0]["lr"],
                )
            )
        scheduler.step()

    print(f"The minimal test loss is {best_loss:.2f} from epoch {best_epoch}")

    if writer:
        writer.flush()
        writer.close()

    return best_loss, best_epoch, train_losses, test_losses, TEST_losses


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
