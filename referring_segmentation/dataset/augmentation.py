import torch
import math
import numbers
import random
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms.functional as F


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        imgs = sample['frames']
        masks = sample['label']
        ims = []
        gts = []
        for i in range(len(sample['frames'])):
            img = imgs[i]
            gt = masks[i]

            if self.padding > 0:
                img = ImageOps.expand(img, border=self.padding, fill=0)
                gt = ImageOps.expand(gt, border=self.padding, fill=0)

            assert img.size == gt.size
            w, h = img.size
            th, tw = self.size # target size
            if w == tw and h == th:
                ims.append(img)
                gts.append(gt)
            elif w < tw or h < th:
                img = img.resize((tw, th), Image.BILINEAR)
                gt = gt.resize((tw, th), Image.NEAREST)
                ims.append(img)
                gts.append(gt)
            else:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                img = img.crop((x1, y1, x1 + tw, y1 + th))
                gt = gt.crop((x1, y1, x1 + tw, y1 + th))
                ims.append(img)
                gts.append(gt)

        return {'frames': ims,
                'label': gts}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        imgs = sample['frames']
        masks = sample['label']
        if random.random() < 0.5:
            ims = []
            gts = []
            for i in range(len(imgs)):
                ims.append(imgs[i].transpose(Image.FLIP_LEFT_RIGHT))
                gts.append(masks[i].transpose(Image.FLIP_LEFT_RIGHT))
            imgs = ims
            masks = gts

        return {'frames': imgs,
                'label': masks}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        ims = []
        gts = []
        for i in range(len(sample['frames'])):

            img = np.array(sample['frames'][i]).astype(np.float32)
            gts.append(np.array(sample['label'][i]).astype(np.float32))

            img /= 255.0
            img -= self.mean
            img /= self.std
            ims.append(img)

        return {'frames': ims,
                'label': gts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        ims = []
        gts = []
        for i in range(len(sample['frames'])):
            img = np.array(sample['frames'][i]).astype(np.float32).transpose((2, 0, 1))
            mask = np.expand_dims(np.array(sample['label'][i]).astype(np.float32), -1).transpose((2, 0, 1))
            mask[mask == 255] = 0

            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
            ims.append(img)
            gts.append(mask)

        return {'frames': ims,
                'label': gts}


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        imgs = sample['frames']
        masks = sample['label']
        ims = []
        gts = []
        for i in range(len(sample['frames'])):

            img = imgs[i].resize(self.size, Image.BILINEAR)
            mask = masks[i].resize(self.size, Image.NEAREST)
            ims.append(img)
            gts.append(mask)

        return {'frames': ims,
                'label': gts}


class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        imgs = sample['frames']
        masks = sample['label']

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * imgs[0].size[0])
        h = int(scale * imgs[0].size[1])
        ims = []
        gts = []
        for i in range(len(sample['frames'])):
            img, mask = imgs[i].resize((w, h), Image.BILINEAR), masks[i].resize((w, h), Image.NEAREST)
            ims.append(img)
            gts.append(mask)

        return {'frames': ims, 'label': gts}


class ExtRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.

        """
        imgs = sample['frames']
        masks = sample['label']
        ims = []
        gts = []
        for i in range(len(imgs)):
            img = imgs[i]
            mask = masks[i]
            if self.padding > 0:
                img = F.pad(img, self.padding)
                mask = F.pad(mask, self.padding)

            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
                mask = F.pad(mask, padding=int((1 + self.size[1] - mask.size[0]) / 2))
            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
                mask = F.pad(mask, padding=int((1 + self.size[0] - mask.size[1]) / 2))

            i, j, h, w = self.get_params(img, self.size)
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            ims.append(img)
            gts.append(mask)

        return  {'frames': ims, 'label': gts}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)