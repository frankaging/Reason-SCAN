import torch
import torch.nn as nn


class ConvolutionalNet(nn.Module):
    """Simple conv. net. Convolves the input channels but retains input image width."""
    def __init__(self, num_channels: int, cnn_kernel_size: int, num_conv_channels: int, dropout_probability: float,
                 stride=1):
        super(ConvolutionalNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=1,
                                padding=0, stride=stride)
        self.conv_2 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=5,
                                stride=stride, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=cnn_kernel_size,
                                stride=stride, padding=cnn_kernel_size // 2)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()
        layers = [self.relu, self.dropout]
        self.layers = nn.Sequential(*layers)
        self.output_dimension = num_conv_channels * 3

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        :param input_images: [batch_size, image_width, image_width, image_channels]
        :return: [batch_size, image_width * image_width, num_conv_channels]
        """
        batch_size = input_images.size(0)
        input_images = input_images.transpose(1, 3)
        conved_1 = self.conv_1(input_images)
        conved_2 = self.conv_2(input_images)
        conved_3 = self.conv_3(input_images)
        images_features = torch.cat([conved_1, conved_2, conved_3], dim=1)
        _, num_channels, _, image_dimension = images_features.size()
        images_features = images_features.transpose(1, 3)
        images_features = self.layers(images_features)
        return images_features.reshape(batch_size, image_dimension * image_dimension, num_channels)


class DeepConvolutionalNet(nn.Module):
    """Simple conv. net. Convolves the input channels but retains input image width."""
    def __init__(self, num_channels: int, num_conv_channels: int, kernel_size: int, dropout_probability: float,
                 stride=1):
        super(DeepConvolutionalNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=1,
                                padding=0, stride=stride)
        self.conv_2 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=3,
                                stride=stride, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=5,
                                stride=stride, padding=2)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()
        layers = [self.relu, self.dropout]
        self.layers = nn.Sequential(*layers)
        self.output_dimension = num_conv_channels * 3

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        :param input_images: [batch_size, image_width, image_width, image_channels]
        :return: [batch_size, image_width * image_width, num_conv_channels]
        """
        batch_size = input_images.size(0)
        input_images = input_images.transpose(1, 3)
        conved_1 = self.conv_1(input_images)
        conved_2 = self.conv_2(input_images)
        conved_3 = self.conv_3(input_images)
        images_features = self.layers(torch.cat([conved_1, conved_2, conved_3], dim=1))
        _, num_channels, _, image_dimension = images_features.size()
        images_features = images_features.transpose(1, 3)
        return images_features.reshape(batch_size, image_dimension * image_dimension, num_channels)


class DownSamplingConvolutionalNet(nn.Module):
    """TODO: make more general and describe"""
    def __init__(self, num_channels: int, num_conv_channels: int, dropout_probability: float):
        super(DownSamplingConvolutionalNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=5,
                                stride=5)
        self.conv_2 = nn.Conv2d(in_channels=num_conv_channels, out_channels=num_conv_channels, kernel_size=3,
                                stride=3, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=num_conv_channels, out_channels=num_conv_channels, kernel_size=3,
                                stride=3, padding=1)
        self.dropout = nn.Dropout2d(dropout_probability)
        self.relu = nn.ReLU()
        layers = [self.conv_1, self.relu, self.dropout, self.conv_2, self.relu, self.dropout, self.conv_3,
                  self.relu, self.dropout]
        self.layers = nn.Sequential(*layers)
        self.output_dimension = num_conv_channels * 3

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        :param input_images: [batch_size, image_width, image_width, image_channels]
        :return: [batch_size, 6 * 6, output_dim]
        """
        batch_size = input_images.size(0)
        input_images = input_images.transpose(1, 3)
        images_features = self.layers(input_images)
        _, num_channels, _, image_dimension = images_features.size()
        images_features = images_features.transpose(1, 3)
        return images_features.reshape(batch_size, image_dimension, image_dimension, num_channels)



