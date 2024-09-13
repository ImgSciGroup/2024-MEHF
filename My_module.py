import torch
from torch import nn


class Unet_Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_Down, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # reduced inception (RI)
        out = self.Down(x)
        return out


class Unet_Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_Up, self).__init__()
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Up(x)


class CSP_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CSP_Block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channel, out_channel, 1)
        self.conv1_1 = nn.Conv2d(in_channel, 256, 1)
        self.conv2_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, out_channel, 1)
        self.conv4_4 = nn.Conv2d(in_channel * 3, out_channel, 1)

    def forward(self, x, y):
        x1 = self.conv0_0(x)
        x2 = self.conv1_1(x)
        x3 = self.conv2_2(x2)
        x4 = self.conv3_3(x3)
        x5 = x1 + x4
        out = torch.cat((x1, x5), dim=1)
        out1 = torch.cat((out, y), dim=1)
        out2 = self.conv4_4(out1)
        return out2


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=4):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         out = self.sigmoid(out)
#         return out
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=3):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv1(out)
#         out = self.sigmoid(out)
#         return out

class Channal_attention(nn.Module):
    def __init__(self, in_ch):
        super(Channal_attention, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_ch // 2, out_channels=in_ch, kernel_size=1), nn.ReLU())
        self.sig = nn.Sigmoid()

    def forward(self, x):
        average = self.conv2(self.conv1(self.ap(x)))
        max = self.conv2(self.conv1(self.mp(x)))
        return x * self.sig(max + average)


class Tensor_attention(nn.Module):
    def __init__(self):
        super(Tensor_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x.permute(0, 2, 1, 3)
        avgx1 = torch.mean(x1, dim=1, keepdim=True)
        maxx1, _ = torch.max(x1, dim=1, keepdim=True)
        out1 = torch.cat([avgx1, maxx1], dim=1)
        out1 = self.conv1(out1)
        out1 = self.sigmoid(out1)
        out1 = out1 * x.permute(0, 2, 1, 3)
        out1 = out1.permute(0, 2, 1, 3)

        x2 = x.permute(0, 3, 2, 1)
        avgx2 = torch.mean(x2, dim=1, keepdim=True)
        maxx2, _ = torch.max(x2, dim=1, keepdim=True)
        out2 = torch.cat([avgx2, maxx2], dim=1)
        out2 = self.conv1(out2)
        out2 = self.sigmoid(out2)
        out2 = out2 * x.permute(0, 3, 2, 1)
        out2 = out2.permute(0, 3, 2, 1)
        return out2 + out1


class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.Channel = Channal_attention(in_ch)
        self.Tensor = Tensor_attention()
        self.convc = nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()

        if (h <= 5):
            x_3d = x
            x_3d_result = (x_3d).view(b, c, h, w)
            cha = self.Channel(x_3d_result)
            return cha
        # average =self.conv2(self.conv1(self.ap(self.conv3d_1 (x.view(b,1,c,h,w)))))
        # max =self.conv2(self.conv1(self.mp(self.conv3d_1 (x.view(b,1,c,h,w)))))
        # x_3d =x.view(b,1,c,h,w) * self.sig(max + average)
        else:
            x_3d = x
            x_3d_result = (x_3d).view(b, c, h, w)
            cha = self.Channel(x_3d_result)
            ten = self.Tensor(cha)
            return ten


class AF(nn.Module):
    def __init__(self):
        super(AF, self).__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(1)
        self.conv = nn.Conv2d(512, 256, 3)
        self.relu = nn.ReLU()
        self.conv0_0 = nn.Conv2d(1024, 512, 1)
        self.conv1_1 = nn.Conv2d(512, 256, 1)
        self.conv2_2 = nn.Conv2d(2048, 1024, 1)

    def forward(self, f1, f2, f3):
        f_3 = self.up1(f3)
        f_up1 = self.relu(self.conv0_0(torch.cat((f2, f_3), dim=1))) # 512
        f_up2 = self.up2(f_up1)
        f_up3 = self.relu(self.conv1_1(torch.cat((f1, f_up2), dim=1)))
        f_down1 = self.down1(f_up3)
        f_down2 = self.relu(self.conv0_0(torch.cat((f_up1, f_down1), dim=1)))
        f_down3 = self.down2(f_down2)
        out = self.relu(self.conv2_2(torch.cat((f3, f_down3), dim=1)))
        return out
class Mutilscale(nn.Module):
    def __init__(self, in_channel):
        super(Mutilscale, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channel, 128, 1)
        self.down0_1 = Unet_Down(128, 256)
        self.down1_2 = Unet_Down(256, 512)
        self.down2_3 = Unet_Down(512, 1024)

        self.up3_0 = Unet_Up(1024, 512)
        self.up2_1 = Unet_Up(512, 256)
        self.up1_2 = Unet_Up(256, 128)

        self.csp1 = CSP_Block(512, 512)
        self.csp2 = CSP_Block(256, 256)
        self.csp3 = CSP_Block(128, 128)

        self.ca1 = CA(256)
        self.ca2 = CA(512)
        self.ca3 = CA(1024)
        self.ca4 = CA(2048)
        # self.diff_spe1 = ChannelAttention(256)
        # self.diff_spe2 = ChannelAttention(512)
        # self.diff_spe3 = ChannelAttention(1024)
        # self.diff_spe4 = ChannelAttention(2048)
        # self.diff_spa = SpatialAttention()
        # self.diff_spe1 = ChannelAttention(256)
        # self.diff_spe2 = ChannelAttention(512)
        # self.diff_spe3 = ChannelAttention(1024)
        self.conv1_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv2_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv3_3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv4_4 = nn.Conv2d(2048, 1024, kernel_size=1)
        # self.conv5_5 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3)
        # self.conv6_6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv5_5 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3)
        self.conv6_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.up1 = Unet_Up(256, 128)
        self.up2 = Unet_Up(512, 256)
        self.up3 = Unet_Up(1024, 512)

        self.af = AF()
        self.down = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(25600, 2048, bias=True),      # NoDFAF 20736
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2, bias=True),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        # x = x.permute(0, 3, 1, 2)
        # y = y.permute(0, 3, 1, 2)
        x = self.conv0_0(x)  # 11  128
        y = self.conv0_0(y)  # 11  128
        x1 = self.down0_1(x)  # 9   256
        x2 = self.down1_2(x1)  # 7  512
        x3 = self.down2_3(x2)  # 5  1024
        x1_1 = self.csp1(x2, self.up3_0(x3))  # 7
        x1_2 = self.csp2(x1, self.up2_1(x1_1))  # 9
        x1_3 = self.csp3(x, self.up1_2(x1_2))  # 11
        y1 = self.down0_1(y)  # 9
        y2 = self.down1_2(y1)  # 7
        y3 = self.down2_3(y2)  # 5
        y1_1 = self.csp1(y2, self.up3_0(y3))  # 7
        y1_2 = self.csp2(y1, self.up2_1(y1_1))  # 9
        y1_3 = self.csp3(y, self.up1_2(y1_2))  # 11
        # out1 = abs(x1_1 - y1_1)  # 512 7 7
        # out2 = abs(x1_2 - y1_2) # 256 9 9
        # out3 = abs(x1_3 - y1_3) # 128 11 11
        # out1 = self.conv5_5(out1)
        # out3 = self.conv6_6(out3)
        # out = out1 + out2 + out3
        cat_11 = torch.cat((x, x1_3), dim=1)
        cat_22 = torch.cat((x1, x1_2), dim=1)
        cat_33 = torch.cat((x2, x1_1), dim=1)
        cat_44 = torch.cat((y, y1_3), dim=1)
        cat_55 = torch.cat((y1, y1_2), dim=1)
        cat_66 = torch.cat((y2, y1_1), dim=1)
        diff_1 = torch.abs(cat_11 - cat_44)  # 9  256
        diff_2 = torch.abs(cat_22 - cat_55)  # 7  512
        diff_3 = torch.abs(cat_33 - cat_66)  # 5  1024
        cat_1 = torch.cat((cat_11, cat_44), dim=1)  # 9 512
        cat_2 = torch.cat((cat_22, cat_55), dim=1)  # 7 1024
        cat_3 = torch.cat((cat_33, cat_66), dim=1)  # 5 2048
        diff_1 = self.ca1(diff_1)
        cat_1 = self.ca2(cat_1)
        cat_1 = self.conv2_2(cat_1)
        out1 = diff_1 + cat_1
        diff_2 = self.ca2(diff_2)
        cat_2 = self.ca3(cat_2)
        cat_2 = self.conv3_3(cat_2)
        out2 = diff_2 + cat_2
        diff_3 = self.ca3(diff_3)
        cat_3 = self.ca4(cat_3)
        cat_3 = self.conv4_4(cat_3)
        out3 = diff_3 + cat_3
        out = self.af(out1, out2, out3)  # d9, d7, d5  256 512 1024
        fe_out = out.clone()
        # out = out1 + out2 + out3
        fe_out = self.down(fe_out)
        out_ = torch.flatten(out, 1, 3)
        out_ = self.fc(out_)
        out_d = self.softmax(out_)
        # final_out = 0.5 * out_c + 0.5 * out_d
        return out_d

