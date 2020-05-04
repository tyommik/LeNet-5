import torch.nn as nn


class C1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(*[
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=(5, 5),
                      stride=1,
                      padding=2),
            nn.LeakyReLU()
        ])

    def forward(self, x):
        x = self.c1(x)
        return x


class S2(nn.Module):
    def __init__(self):
        super().__init__()
        self.s2 = nn.AvgPool2d((2, 2), stride=2)

    def forward(self, x):
        x = self.s2(x)
        return x


class C3(nn.Module):
    def __init__(self):
        super().__init__()
        self.c3 = nn.Sequential(*[
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=(5, 5),
                      stride=1,
                      padding=0),
            nn.LeakyReLU()
        ])

    def forward(self, x):
        x = self.c3(x)
        return x


class S4(nn.Module):
    def __init__(self):
        super().__init__()
        self.s4 = nn.AvgPool2d((2, 2), stride=2)

    def forward(self, x):
        x = self.s4(x)
        return x


class C5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c5 = nn.Sequential(*[
            nn.Linear(in_features=5 * 5 * 16, out_features=120),
            nn.Tanh()
        ])

    def forward(self, x):
        x = self.c5(x)
        return x


class F6(nn.Module):
    def __init__(self):
        super().__init__()
        self.f6 = nn.Sequential(*[
            nn.Linear(in_features=120, out_features=84),
            nn.LeakyReLU()
        ])

    def forward(self, x):
        x = self.f6(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = C1()
        self.s2 = S2()
        self.c3 = C3()
        self.s4 = S4()
        self.c5 = C5()
        self.f6 = F6()

        self.output = nn.Linear(in_features=84, out_features=10)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = x.reshape(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.c5(x)
        x = self.f6(x)
        x = self.output(x)

        return x

    def inference(self, x):
        x = self.sm(x)
        return x