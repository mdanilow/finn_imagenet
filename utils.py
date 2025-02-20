from os.path import join
import shutil
from pathlib import Path
from copy import deepcopy
import math
from multiprocessing.pool import ThreadPool

import glob
import re
import torch
from torch import nn
from tqdm import tqdm
import tarfile
from PIL import Image
import io

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset, find_classes, IMG_EXTENSIONS, default_loader


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
    

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    torch.save(state, join(save_dir, filename))
    if is_best:
        shutil.copyfile(join(save_dir, filename), join(save_dir, 'model_best.pth.tar'))


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0, average_quant_scales=False):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.device = next(model.parameters()).device
        print('ema device:', self.device)
        # self.ema.to(self.device)
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.overwrite_quant_scales = not average_quant_scales
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            new_keys = False
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            ema_msd = self.ema.state_dict()
            for k, v in self.ema.state_dict().items():
                # for Brevitas activation scales, keep the newest value
                if self.overwrite_quant_scales and '.tensor_quant.scaling_impl.value' in k:
                    scale = 0
                else:
                    scale = d
                if v.dtype.is_floating_point:
                    v = v.to(self.device)
                    v *= scale
                    v += (1. - scale) * msd[k].detach()
            for k, v in msd.items():
                if k not in ema_msd:
                    new_keys = True
                    print('adding', k, 'to ema')
                    ema_msd[k] = v
            if new_keys:
                print('LOADED')
                self.ema.load_state_dict(ema_msd)


class MyDatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        cache_images: bool = False,
        from_tar: bool = False
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        # caches images by default, for train ImageNet only
        if from_tar:
            self.cache_images = True
            self.imgs = []
            classes = []
            samples = []
            # read metadata
            with open(self.root, 'rb') as fd:
                with tarfile.open(fileobj=fd, mode='r') as tar_root:
                    for item in tar_root:
                        classes.append(item.name)
            classes = sorted(classes)
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

            with open(self.root, 'rb') as fd:
                with tarfile.open(fileobj=fd, mode='r') as tar_root:
                    pbar = tqdm(tar_root, total=len(classes))
                    # enumerate_pbar = enumerate(pbar)
                    pbar.desc = "Caching train images"
                    for item in pbar:
                        with tar_root.extractfile(item) as class_fd:
                            with tarfile.open(fileobj=class_fd, mode='r') as class_tar:
                                for i, img_file in enumerate(class_tar):
                                    img = class_tar.extractfile(img_file)
                                    img = img.read()
                                    img = Image.open(io.BytesIO(img)).convert("RGB")
                                    self.imgs.append(img)
                                    samples.append((None, class_to_idx[item.name]))
            
        else:
            classes, class_to_idx = self.find_classes(self.root)
            samples = self.make_dataset(
                self.root,
                class_to_idx=class_to_idx,
                extensions=extensions,
                is_valid_file=is_valid_file,
                allow_empty=allow_empty,
            )

        self.cache_images = cache_images
        self.from_tar = from_tar
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if cache_images and not from_tar:
            self.imgs = [None] * len(self.samples)
            gb = 0
            results = ThreadPool(8).imap(lambda x: self.loader(x[0]), self.samples)
            pbar = tqdm(enumerate(results), total=len(self.samples))
            for i, x in pbar:
                self.imgs[i] = x
                # gb += self.imgs[i].nbytes
                # pbar.desc = f'Caching images ({gb / 1E9:.1f}GB)'
                pbar.desc = 'Caching val images'

    @staticmethod
    def make_dataset(
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
            allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
                An error is raised on empty folders if False (default).

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
        )

    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        if self.cache_images:
            target = self.samples[index][1]
            sample = self.imgs[index]
        else:
            path, target = self.samples[index]
            sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class MyImageFolder(MyDatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        cache_images: bool = False,
        from_tar: bool = False
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
            cache_images=cache_images,
            from_tar=from_tar
        )
        # self.imgs = self.samples


