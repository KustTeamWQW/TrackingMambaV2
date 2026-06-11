import random
import numpy as np
import math
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf


class Transform:
    """A set of transformations, used for e.g. data augmentation.
    Args of constructor:
        transforms: An arbitrary number of transformations, derived from the TransformBase class.
                    They are applied in the order they are given.

    The Transform object can jointly transform images, bounding boxes and segmentation masks.
    This is done by calling the object with the following key-word arguments (all are optional).

    The following arguments are inputs to be transformed. They are either supplied as a single instance, or a list of instances.
        image  -  Image
        coords  -  2xN dimensional Tensor of 2D image coordinates [y, x]
        bbox  -  Bounding box on the form [x, y, w, h]
        mask  -  Segmentation mask with discrete classes

    The following parameters can be supplied with calling the transform object:
        joint [Bool]  -  If True then transform all images/coords/bbox/mask in the list jointly using the same transformation.
                         Otherwise each tuple (images, coords, bbox, mask) will be transformed independently using
                         different random rolls. Default: True.
        new_roll [Bool]  -  If False, then no new random roll is performed, and the saved result from the previous roll
                            is used instead. Default: True.

    Check the DiMPProcessing class for examples.
    """

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['joint', 'new_roll']
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        for v in inputs.keys():
            if v not in self._valid_all:
                raise ValueError('Incorrect input \"{}\" to transform. Only supports inputs {} and arguments {}.'.format(v, self._valid_inputs, self._valid_args))

        joint_mode = inputs.get('joint', True)
        new_roll = inputs.get('new_roll', True)

        if not joint_mode:
            out = zip(*[self(**inp) for inp in self._split_inputs(inputs)])
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll)
        if len(var_names) == 1:
            return out[var_names[0]]
        # Make sure order is correct
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [{k: v for k, v in zip(var_names, vals)} for vals in zip(*[inputs[vn] for vn in var_names])]
        for arg_name, arg_val in filter(lambda it: it[0]!='joint' and it[0] in self._valid_args, inputs.items()):
            if isinstance(arg_val, list):
                for inp, av in zip(split_inputs, arg_val):
                    inp[arg_name] = av
            else:
                for inp in split_inputs:
                    inp[arg_name] = arg_val
        return split_inputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TransformBase:
    """Base class for transformation objects. See the Transform class for details."""
    def __init__(self):
        """2020.12.24 Add 'att' to valid inputs"""
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['new_roll']
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        # Split input
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        # Roll random parameters for the transform
        if input_args.get('new_roll', True):
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = dict()
        for var_name, var in input_vars.items():
            if var is not None:
                transform_func = getattr(self, 'transform_' + var_name)
                if var_name in ['coords', 'bbox']:
                    params = (self._get_image_size(input_vars),) + self._rand_params
                else:
                    params = self._rand_params
                if isinstance(var, (list, tuple)):
                    outputs[var_name] = [transform_func(x, *params) for x in var]
                else:
                    outputs[var_name] = transform_func(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        im = None
        for var_name in ['image', 'mask']:
            if inputs.get(var_name) is not None:
                im = inputs[var_name]
                break
        if im is None:
            return None
        if isinstance(im, (list, tuple)):
            im = im[0]
        if isinstance(im, np.ndarray):
            return im.shape[:2]
        if torch.is_tensor(im):
            return (im.shape[-2], im.shape[-1])
        raise Exception('Unknown image type')

    def roll(self):
        return None

    def transform_image(self, image, *rand_params):
        """Must be deterministic"""
        return image

    def transform_coords(self, coords, image_shape, *rand_params):
        """Must be deterministic"""
        return coords

    def transform_bbox(self, bbox, image_shape, *rand_params):
        """Assumes [x, y, w, h]"""
        # Check if not overloaded
        if self.transform_coords.__code__ == TransformBase.transform_coords.__code__:
            return bbox

        coord = bbox.clone().view(-1,2).t().flip(0)

        x1 = coord[1, 0]
        x2 = coord[1, 0] + coord[1, 1]

        y1 = coord[0, 0]
        y2 = coord[0, 0] + coord[0, 1]

        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])

        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        bbox_out = torch.cat((tl, sz), dim=-1).reshape(bbox.shape)
        return bbox_out

    def transform_mask(self, mask, *rand_params):
        """Must be deterministic"""
        return mask

    def transform_att(self, att, *rand_params):
        """2020.12.24 Added to deal with attention masks"""
        return att


class ToTensor(TransformBase):
    """Convert to a Tensor"""

    def transform_image(self, image):
        # handle numpy array
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        else:
            return image

    def transfrom_mask(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)

    def transform_att(self, att):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class ToTensorAndJitter(TransformBase):
    """Convert to a Tensor and jitter brightness"""
    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform_image(self, image, brightness_factor):
        # handle numpy array
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # backward compatibility
        if self.normalize:
            return image.float().mul(brightness_factor/255.0).clamp(0.0, 1.0)
        else:
            return image.float().mul(brightness_factor).clamp(0.0, 255.0)

    def transform_mask(self, mask, brightness_factor):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        else:
            return mask
    def transform_att(self, att, brightness_factor):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError ("dtype must be np.ndarray or torch.Tensor")


class MountainJungleAugment(TransformBase):
    """Tensor image perturbations for UAV mountain-jungle tracking."""
    def __init__(self, probability=0.65, occlusion_prob=0.45, max_occlusions=3,
                 low_light_prob=0.35, haze_prob=0.25, blur_prob=0.25,
                 noise_prob=0.30, color_prob=0.45):
        super().__init__()
        self.probability = probability
        self.occlusion_prob = occlusion_prob
        self.max_occlusions = max_occlusions
        self.low_light_prob = low_light_prob
        self.haze_prob = haze_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.color_prob = color_prob

    def roll(self):
        params = {"apply": random.random() < self.probability}
        if not params["apply"]:
            return params

        params["low_light"] = random.random() < self.low_light_prob
        params["haze"] = random.random() < self.haze_prob
        params["blur"] = random.random() < self.blur_prob
        params["noise"] = random.random() < self.noise_prob
        params["color"] = random.random() < self.color_prob

        params["gamma"] = random.uniform(0.65, 1.75)
        params["gain"] = random.uniform(0.60, 1.15)
        params["contrast"] = random.uniform(0.75, 1.35)
        params["saturation"] = random.uniform(0.70, 1.25)
        params["channel_scale"] = [random.uniform(0.88, 1.12) for _ in range(3)]
        params["haze_strength"] = random.uniform(0.08, 0.32)
        params["airlight"] = random.uniform(0.70, 0.96)
        params["noise_std"] = random.uniform(0.006, 0.035)
        params["blur_kernel"] = random.choice([3, 5])
        params["blur_horizontal"] = random.random() < 0.5

        occlusions = []
        if random.random() < self.occlusion_prob:
            num_occ = random.randint(1, self.max_occlusions)
            foliage_colors = [
                (0.08, 0.22, 0.07),
                (0.13, 0.32, 0.10),
                (0.24, 0.26, 0.12),
                (0.34, 0.23, 0.13),
            ]
            for _ in range(num_occ):
                base_color = random.choice(foliage_colors)
                occlusions.append({
                    "cx": random.random(),
                    "cy": random.random(),
                    "w": random.uniform(0.06, 0.28),
                    "h": random.uniform(0.04, 0.22),
                    "opacity": random.uniform(0.45, 0.85),
                    "texture": random.uniform(0.02, 0.09),
                    "color": base_color,
                })
        params["occlusions"] = occlusions
        return params

    def transform_image(self, image, params=None):
        if params is None or not params.get("apply", False):
            return image
        if not torch.is_tensor(image):
            return image

        out = image.clone().float()
        original_dtype = image.dtype

        if params.get("low_light", False):
            out = out.clamp(0.0, 1.0).pow(params["gamma"]).mul(params["gain"])

        if params.get("color", False) and out.shape[0] == 3:
            mean = out.mean(dim=(1, 2), keepdim=True)
            gray = (out[0:1] * 0.2989 + out[1:2] * 0.5870 + out[2:3] * 0.1140)
            out = (out - mean) * params["contrast"] + mean
            out = gray + (out - gray) * params["saturation"]
            scale = torch.tensor(params["channel_scale"], dtype=out.dtype, device=out.device).view(3, 1, 1)
            out = out * scale

        if params.get("haze", False):
            strength = params["haze_strength"]
            out = out * (1.0 - strength) + params["airlight"] * strength

        for occ in params.get("occlusions", []):
            _, h, w = out.shape
            occ_w = max(2, int(w * occ["w"]))
            occ_h = max(2, int(h * occ["h"]))
            cx = int(occ["cx"] * w)
            cy = int(occ["cy"] * h)
            x0 = max(0, min(w - 1, cx - occ_w // 2))
            y0 = max(0, min(h - 1, cy - occ_h // 2))
            x1 = max(x0 + 1, min(w, x0 + occ_w))
            y1 = max(y0 + 1, min(h, y0 + occ_h))
            color = torch.tensor(occ["color"], dtype=out.dtype, device=out.device).view(-1, 1, 1)
            if color.shape[0] != out.shape[0]:
                color = color[:1].expand(out.shape[0], 1, 1)
            texture = torch.randn((out.shape[0], y1 - y0, x1 - x0), dtype=out.dtype,
                                  device=out.device) * occ["texture"]
            patch = (color + texture).clamp(0.0, 1.0)
            out[:, y0:y1, x0:x1] = out[:, y0:y1, x0:x1] * (1.0 - occ["opacity"]) + patch * occ["opacity"]

        if params.get("blur", False):
            out = self._motion_blur(out, params["blur_kernel"], params["blur_horizontal"])

        if params.get("noise", False):
            out = out + torch.randn_like(out) * params["noise_std"]

        return out.clamp(0.0, 1.0).to(dtype=original_dtype)

    @staticmethod
    def _motion_blur(image, kernel_size, horizontal):
        c, _, _ = image.shape
        kernel = torch.zeros((c, 1, kernel_size, kernel_size), dtype=image.dtype, device=image.device)
        center = kernel_size // 2
        if horizontal:
            kernel[:, 0, center, :] = 1.0 / kernel_size
        else:
            kernel[:, 0, :, center] = 1.0 / kernel_size
        return F.conv2d(image.unsqueeze(0), kernel, padding=center, groups=c).squeeze(0)


class Normalize(TransformBase):
    """Normalize image"""
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, image):
        return tvisf.normalize(image, self.mean, self.std, self.inplace)


class ToGrayscale(TransformBase):
    """Converts image to grayscale with probability"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_grayscale):
        if do_grayscale:
            if torch.is_tensor(image):
                raise NotImplementedError('Implement torch variant.')
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2)
            # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
        return image


class ToBGR(TransformBase):
    """Converts image to BGR"""
    def transform_image(self, image):
        if torch.is_tensor(image):
            raise NotImplementedError('Implement torch variant.')
        img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img_bgr


class RandomHorizontalFlip(TransformBase):
    """Horizontally flip image randomly with a probability p."""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((2,))
            return np.fliplr(image).copy()
        return image

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = (image_shape[1] - 1) - coords[1,:]
            return coords_flip
        return coords

    def transform_mask(self, mask, do_flip):
        if do_flip:
            if torch.is_tensor(mask):
                return mask.flip((-1,))
            return np.fliplr(mask).copy()
        return mask

    def transform_att(self, att, do_flip):
        if do_flip:
            if torch.is_tensor(att):
                return att.flip((-1,))
            return np.fliplr(att).copy()
        return att


class RandomHorizontalFlip_Norm(RandomHorizontalFlip):
    """Horizontally flip image randomly with a probability p.
    The difference is that the coord is normalized to [0,1]"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def transform_coords(self, coords, image_shape, do_flip):
        """we should use 1 rather than image_shape"""
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1,:] = 1 - coords[1,:]
            return coords_flip
        return coords
