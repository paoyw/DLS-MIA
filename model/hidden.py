from io import BytesIO
from random import choice

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision.transforms.functional as TFF
from torchvision.utils import save_image


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class HiddenEncoder(nn.Module):
    def __init__(self, channels=64, num_blocks=4, message_length=32):
        super().__init__()
        self.conv_channels = channels
        self.num_blocks = num_blocks
        self.message_length = message_length

        layers = [ConvBNReLU(3, channels)]

        for _ in range(num_blocks - 1):
            layers.append(ConvBNReLU(channels, channels))

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNReLU(
            message_length + 3 + channels, channels
        )
        self.last_layer = nn.Conv2d(
            in_channels=channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, images: torch.Tensor, messages: torch.Tensor):
        if images.dim() != 4 or \
                images.shape[1] != 3 or \
                messages.dim() != 2 or \
                messages.shape[1] != self.message_length or \
                images.shape[0] != messages.shape[0]:
            raise ValueError('The shape of the images should be B x 3 x H x W.'
                             'The shape of the messages should be B x L.')

        batch_size, _, height, width = images.shape
        expand_messages = messages \
            .view(batch_size, self.message_length, 1, 1)\
            .expand(batch_size,
                    self.message_length,
                    height,
                    width)

        encoded_images = self.conv_layers(images)
        concat = torch.concat((expand_messages, encoded_images, images), dim=1)
        act = self.after_concat_layer(concat)
        return self.last_layer(act)


class HiddenDecoder(nn.Module):
    def __init__(self, channels=64, num_blocks=7, message_length=32):
        super().__init__()
        self.conv_channels = channels
        self.num_blocks = num_blocks
        self.message_length = message_length

        layers = [ConvBNReLU(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNReLU(channels, channels))
        layers.append(ConvBNReLU(channels, message_length))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.conv_layers = nn.Sequential(*layers)

        self.linear = nn.Linear(message_length, message_length)

    def forward(self, images: torch.Tensor):
        if images.dim() != 4 or \
                images.shape[1] != 3:
            raise ValueError('The shape of the images should be'
                             'B x 3 x H x W.')

        x = self.conv_layers(images).view(images.shape[0], self.message_length)
        return self.linear(x)


class HiddenAdversary(nn.Module):
    def __init__(self, channels=64, num_blocks=3):
        super().__init__()
        self.conv_channels = channels
        self.num_blocks = num_blocks

        layers = [ConvBNReLU(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNReLU(channels, channels))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.conv_layers = nn.Sequential(*layers)

        self.linear = nn.Linear(channels, 2)

    def forward(self, images: torch.Tensor):
        if images.dim() != 4 or \
                images.shape[1] != 3:
            raise ValueError('The shape of the images should be'
                             'B x 3 x H x W.')

        x = self.conv_layers(images)
        x = x.squeeze(-1).squeeze(-1)
        return self.linear(x)


class ApproximateJPEG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quality):
        ctx.save_for_backward(x)
        if x.dim() == 3:
            device = x.device
            buffer = BytesIO()
            im = TFF.to_pil_image(x.cpu())
            im.save(buffer, 'jpeg', quality=quality)
            im = Image.open(buffer)
            return TFF.to_tensor(im).to(device)
        elif x.dim() == 4:
            device = x.device
            ims = []
            for im in x:
                buffer = BytesIO()
                im = TFF.to_pil_image(im.to('cpu'))
                im.save(buffer, 'jpeg', quality=quality)
                im = Image.open(buffer)
                im = TFF.to_tensor(im)
                ims.append(im)
            return torch.stack(ims).to(device)
        else:
            raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs, = ctx.saved_tensors
        return grad_outputs, None


def jpeg_compress(x, mean=0.5, std=0.5, quality=80):
    x_denorm = (std * x + mean).clip(0, 1)
    x_jpeg = ApproximateJPEG.apply(x_denorm, quality)
    return (x_jpeg - mean) / std


class JPEGCompress(nn.Module):

    def __init__(self, quality: int):
        super().__init__()
        self.quality = quality

    def __str__(self) -> str:
        return f'JPEGCompress(quality={self.quality})'

    def forward(self, x, mean=0.5, std=0.5):
        return jpeg_compress(x, mean, std, self.quality)


def cropout(x, size):
    if isinstance(size, int):
        th = size
        tw = size
    elif isinstance(size, list):
        th = size[0]
        tw = size[1]
    else:
        raise TypeError
    h, w = x.shape[-2], x.shape[-1]
    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    response = torch.randn_like(x) * x.std() + x.mean()

    if x.dim() == 3:
        response[:, i:i + th, j:j + tw] = \
            x[:, i:i + th, j:j + tw]
    elif x.dim() == 4:
        response[:, :, i:i + th, j:j + tw] = \
            x[:, :, i:i + th, j:j + tw]
    else:
        raise ValueError
    return response


class Cropout(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x, size=None):
        if size is None:
            size = self.size
        return cropout(x, size)


class HiddenNoisysLayer(nn.Module):
    def __init__(self, size=256):
        super().__init__()
        self.size = size

    def forward(self, images: torch.Tensor, size=None, dropout_p=0.3, crop_ratio=0.7, cropout_ratio=0.035, jpeg_quality=50):
        if images.dim() != 4 or \
                images.shape[1] != 3:
            raise ValueError('The shape of the images should be'
                             'B x 3 x H x W.')

        if size is None:
            size = self.size

        tfms = [
            lambda x: x,
            # nn.Dropout(p=dropout_p),
            # transforms.Compose([
            #     transforms.RandomCrop(int(size * (crop_ratio ** 0.5))),
            #     transforms.Resize(size)
            # ]),
            # Cropout(size=int(size * (cropout_ratio ** 0.5))),
            # transforms.GaussianBlur(5),
            # JPEGCompress(jpeg_quality)
        ]

        return torch.stack([choice(tfms)(image) for image in images])


class Hidden(nn.Module):
    def __init__(self,
                 encoder: HiddenEncoder,
                 decoder: HiddenDecoder,
                 noisy_layer: HiddenNoisysLayer,
                 adversary: HiddenAdversary):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noisy_layer = noisy_layer
        self.adversary = adversary
