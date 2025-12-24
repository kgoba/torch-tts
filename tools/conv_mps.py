import torch
import torch.nn as nn

class Conv1dFix(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, bias=True):
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros([out_channels, in_channels, kernel_size]))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros([out_channels]))
            nn.init.normal_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, self.padding))  # N, C, L
        x = torch.cat(
            [x.roll(n - self.padding, dims=2) for n in range(self.kernel_size)], dim=1
        )
        x = x[:, :, self.padding : -self.padding]
        x = torch.matmul(self.weight.view(self.out_channels, -1), x)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(-1)
        return x


def main():
    # self.conv_weight = nn.Parameter(torch.zeros([dim_in, dim_in, 5]))
    # self.conv_bias = nn.Parameter(torch.zeros([dim_in]))
    # nn.init.normal_(self.conv_weight, 0, 1/dim_in)
    # nn.init.normal_(self.conv_bias)
    # x = nn.functional.conv1d(nn.functional.pad(x.mT, (2, 2)), self.conv_weight, self.conv_bias).mT

    N = 1
    C = 2
    L = 16
    x = torch.zeros((N, C, L), requires_grad=True)
    nn.init.normal_(x)

    conv = nn.Conv1d(C, C, kernel_size=5, padding=2)
    nn.init.ones_(conv.weight)
    conv_mps = Conv1dFix(C, C, kernel_size=5, padding=2)
    conv_mps.weight = torch.nn.Parameter(conv.weight.clone())
    conv_mps.bias = torch.nn.Parameter(conv.bias.clone())
    conv_mps.to('mps')

    y_cpu = conv(x)
    print('CPU:', y_cpu)
    # y_cpu.retain_grad()
    # torch.sum(y_cpu).backward(retain_graph=True)
    print('CPU:', y_cpu.grad_fn(x))

    x_mps = x.to('mps')
    y_mps = conv_mps(x_mps)
    print('MPS:', y_mps)
    # y_mps.retain_grad()
    # torch.sum(y_mps).backward(retain_graph=True)
    print('MPS:', y_mps.grad_fn(x_mps))


if __name__=='__main__':
    main()
