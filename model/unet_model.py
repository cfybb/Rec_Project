"""
UNet Implementation from https://github.com/milesial/Pytorch-UNet/blob/master/unet, with minor adjustments.
Full assembly of the parts to form the complete network
"""

# from models.unet_parts import *
from unet_parts import *
DOWNSAMPLE_NUM = 4
UPSAMPLE_NUM = 2


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.recorder = []
        self.factor = 0

        self.inc = (DoubleConv(n_channels, 64))
        # self.down1 = (Down(64, 128))
        # self.down2 = (Down(128, 256))
        # self.down3 = (Down(256, 512))
        # factor = 2 if bilinear else 1
        # self.down4 = (Down(512, 1024 // factor))
        # self.up1 = (Up(1024, 512 // factor, bilinear))
        # self.up2 = (Up(512, 256 // factor, bilinear))
        # self.up3 = (Up(256, 128 // factor, bilinear))
        # self.up4 = (Up(128, 64, bilinear))
        # self.outc = (OutConv(64, n_classes))
        self.encoder = self._make_downsample_layers(64, DOWNSAMPLE_NUM)
        self.decoder = self._make_upsample_layers(in_channels=64*(2**DOWNSAMPLE_NUM), num_layers=UPSAMPLE_NUM)
        self.outc = OutConv(64*2**(UPSAMPLE_NUM), n_classes)
        # TODO: add a sigmoid layer to apply (0, 1) constraint to the output
        self.sigmoid = nn.Sigmoid()

        assert DOWNSAMPLE_NUM >= UPSAMPLE_NUM
        # TODO: use DOWNSAMPLE_NUM and UPSAMPLE_NUM to customize model architecture.

    def _make_downsample_layers(self, in_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            self.factor = 2 if self.bilinear and num_layers == DOWNSAMPLE_NUM else 1
            layers.append(Down(in_channels, in_channels*2 // self.factor))
            in_channels = in_channels*2//self.factor
            print("in_channels for down",in_channels)
        return nn.Sequential(*layers)

    def _make_upsample_layers(self, in_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            print("in_channels for up",in_channels)
            # print("self factor for up",self.factor)
            layers.append(Up(in_channels, (in_channels // 2)//self.factor, self.bilinear))
            in_channels = (in_channels//2)//self.factor
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.inc(x)
        for id, model in enumerate(self.encoder):
            x = model(x)
            if id < DOWNSAMPLE_NUM -1 :
                self.recorder.append(x)
        # forward will only run once, but to make sure the counter is reset properly, initialization will be here.
        # for record in self.recorder:
        #    print("recorder:",record.shape)
        for i, model_d in enumerate(self.decoder):
            print(len(self.recorder))
            downsample_result = self.recorder[DOWNSAMPLE_NUM - i -2 ]
            x = model_d(x, downsample_result)
        x = self.outc(x)
        logits = self.sigmoid(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.encoder = torch.utils.checkpoint.checkpoint(self.encoder)
        self.decoder = torch.utils.checkpoint.checkpoint(self.decoder)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)


if __name__ == "__main__":
    unet = UNet(n_channels=3, n_classes=8)
    #print(unet)
    input = torch.rand([1, 3, 720, 1280], dtype=torch.float32)
    print("input shape", input.shape)
    output = unet(input)
    print(output.shape)  # [1, 8, 720 * 2 ^ (UPSAMPLE - DOWNSAMPLE), 1280 * 2 ^ (UPSAMPLE - DOWNSAMPLE)]
