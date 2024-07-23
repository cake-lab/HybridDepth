import pytorch_lightning as pl
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader, random_split
import torch
from typing import Tuple
from dataloader.DDFF12.loader import DDFF12Loader
from dataloader.FOD500.loader import FoD500Loader
from .NYU.loader import NYULoader
from .ToF18.loader import ToFLoader
from dataloader.DIODE.loader import DIODELoader
from dataloader.ARKitScenes.loader import ARKitScenesLoader
from dataloader.Matterport3D.loader import Matterport3DLoader


class FOD500DDFF12DataModule(pl.LightningDataModule):
    def __init__(
        self,
        ddff12_root: str = "",
        fod500_root: str = "",
        img_size: Tuple = (224, 224),
        batch_size: int = 32,
        num_workers: int = 16,
        use_labels: bool = True,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        self.ddff12database_root = ddff12_root
        self.fod500database_root = fod500_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_labels = use_labels
        self.num_cluster = num_cluster

        self.focus_dist = [0.1, .15, .3, 0.7, 1.5]
        self.img_train_list = [f for f in listdir(self.fod500database_root) if isfile(
            join(self.fod500database_root, f)) and f[-7:] == "All.tif" and int(f[:6]) < 400]
        self.dpth_train_list = [f for f in listdir(self.fod500database_root) if isfile(
            join(self.fod500database_root, f)) and f[-7:] == "Dpt.exr" and int(f[:6]) < 400]

        self.img_train_list.sort()
        self.dpth_train_list.sort()

        self.img_val_list = [f for f in listdir(self.fod500database_root) if isfile(
            join(self.fod500database_root, f)) and f[-7:] == "All.tif" and int(f[:6]) >= 400]
        self.dpth_val_list = [f for f in listdir(self.fod500database_root) if isfile(
            join(self.fod500database_root, f)) and f[-7:] == "Dpt.exr" and int(f[:6]) >= 400]

        self.img_val_list.sort()
        self.dpth_val_list.sort()

    def setup(self, stage: str):
        if stage == "fit":
            # Create the DDFF12 dataset
            ddff12_dataset = DDFF12Loader(
                self.ddff12database_root,
                stack_key="stack_train",
                disp_key="disp_train",
                n_stack=10,
                min_disp=0.02,
                max_disp=0.28,
            )
            # Create the FoD500 dataset
            fod500_dataset = FoD500Loader(
                self.fod500database_root,
                stage="train",
                img_list=self.img_train_list,
                dpth_list=self.dpth_train_list,
                img_num=10,
                focus_dist=self.focus_dist,
                scale=0.2
            )
            # Concatenate the DDFF12 and FoD500 datasets
            self.train_dataset = torch.utils.data.ConcatDataset(
                [ddff12_dataset, fod500_dataset])

            self.valid_dataset = DDFF12Loader(
                self.ddff12database_root,
                stack_key="stack_val",
                disp_key="disp_val",
                n_stack=10,
                min_disp=0.02,
                max_disp=0.28,
                b_test=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=12, num_workers=self.num_workers
        )


class DDFF12DataModule(pl.LightningDataModule):
    def __init__(
        self,
        ddff12_data_root: str = "",
        img_size: Tuple = (480, 640),
        remove_white_border: bool = True,
        batch_size: int = 32,
        num_workers: int = 16,
        use_labels: bool = True,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        self.database_root = ddff12_data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_labels = use_labels
        self.num_cluster = num_cluster

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = DDFF12Loader(
                self.database_root,
                stack_key="stack_train",
                disp_key="disp_train",
                n_stack=5,
                min_disp=0.02,
                max_disp=0.28,
            )
            self.valid_dataset = DDFF12Loader(
                self.database_root,
                stack_key="stack_val",
                disp_key="disp_val",
                n_stack=5,
                min_disp=0.02,
                max_disp=0.28,
                b_test=True,
            )
        if stage == "test":
            self.valid_dataset = DDFF12Loader(
                self.database_root,
                stack_key="stack_val",
                disp_key="disp_val",
                n_stack=5,
                min_disp=0.02,
                max_disp=0.28,
                b_test=True,
            )
            self.test_dataset = self.valid_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)


class FOD500DataModule(pl.LightningDataModule):
    def __init__(
        self,
        fod500_root: str = "",
        batch_size: int = 32,
        num_workers: int = 16,
        use_labels: bool = True,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        self.database_root = fod500_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_labels = use_labels
        self.num_cluster = num_cluster

        self.focus_dist = [0.1, .15, .3, 0.7, 1.5]
        self.img_train_list = [f for f in listdir(self.database_root) if isfile(
            join(self.database_root, f)) and f[-7:] == "All.tif" and int(f[:6]) < 400]
        self.dpth_train_list = [f for f in listdir(self.database_root) if isfile(
            join(self.database_root, f)) and f[-7:] == "Dpt.exr" and int(f[:6]) < 400]

        self.img_train_list.sort()
        self.dpth_train_list.sort()

        self.img_val_list = [f for f in listdir(self.database_root) if isfile(
            join(self.database_root, f)) and f[-7:] == "All.tif" and int(f[:6]) >= 400]
        self.dpth_val_list = [f for f in listdir(self.database_root) if isfile(
            join(self.database_root, f)) and f[-7:] == "Dpt.exr" and int(f[:6]) >= 400]

        self.img_val_list.sort()
        self.dpth_val_list.sort()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = FoD500Loader(
                self.database_root,
                stage="train",
                img_list=self.img_train_list, dpth_list=self.dpth_train_list,
                img_num=5, focus_dist=self.focus_dist, scale=0.2
            )
            self.valid_dataset = FoD500Loader(
                self.database_root,
                img_list=self.img_val_list, dpth_list=self.dpth_val_list,
                img_num=5, focus_dist=self.focus_dist, scale=0.2,
                stage="valid"
            )
        if stage == "test":
            self.test_dataset = FoD500Loader(
                self.database_root,
                img_list=self.img_val_list, dpth_list=self.dpth_val_list,
                img_num=5, focus_dist=self.focus_dist, scale=1,
                stage="valid"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=8, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)


class NYUDataModule(pl.LightningDataModule):
    def __init__(
        self,
        nyuv2_data_root: str = "",
        img_size: Tuple = (480, 640),
        remove_white_border: bool = True,
        batch_size: int = 32,
        num_workers: int = 16,
        use_labels: bool = True,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        # Set the seed for PyTorch
        # torch.manual_seed(42)
        self.nyu_data_root = nyuv2_data_root
        self.image_size = img_size
        self.remove_white_border = remove_white_border

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_labels = use_labels
        self.num_cluster = num_cluster

        # dataset_train, dataset_valid = random_split(dataset, [0.9, 0.1])

        # self.train_dataset = dataset_train
        # self.valid_dataset = dataset_valid
        # self.test_dataset = dataset_valid

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = NYULoader(
                self.nyu_data_root,
                img_size=self.image_size,
                remove_white_border=self.remove_white_border,
                stage="train"
            )
            self.test_dataset = NYULoader(
                self.nyu_data_root,
                img_size=self.image_size,
                remove_white_border=self.remove_white_border,
                stage="test"
            )
        if stage == "test":
            self.test_dataset = NYULoader(
                self.nyu_data_root,
                img_size=self.image_size,
                remove_white_border=self.remove_white_border,
                stage="test"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)


class ARKitScenesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        arkitScenes_data_root: str = "",
        img_size: Tuple = (192, 256),
        batch_size: int = 32,
        num_workers: int = 16,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        # Set the seed for PyTorch
        # torch.manual_seed(42)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_cluster = num_cluster

        self.arkitScenes_data_root = arkitScenes_data_root
        self.image_size = img_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = ARKitScenesLoader(
                ARKitScenes_data_root=self.arkitScenes_data_root,
                img_size=self.image_size,
                stage="Training"
            )
            self.valid_dataset = ARKitScenesLoader(
                ARKitScenes_data_root=self.arkitScenes_data_root,
                img_size=self.image_size,
                stage="Validation"
            )
        if stage == "test":
            self.test_dataset = ARKitScenesLoader(
                ARKitScenes_data_root=self.arkitScenes_data_root,
                img_size=self.image_size,
                stage="Validation"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)


class Matterport3dDataModule(pl.LightningDataModule):
    def __init__(
        self,
        matterport3d_data_root: str = "",
        img_size: Tuple = (192, 256),
        batch_size: int = 32,
        num_workers: int = 16,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        # Set the seed for PyTorch
        # torch.manual_seed(42)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_cluster = num_cluster

        self.matterport3d_data_root = matterport3d_data_root
        self.image_size = img_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = Matterport3DLoader(
                Matterport3D_root=self.matterport3d_data_root,
                img_size=self.image_size,
                stage="train"
            )
            self.valid_dataset = Matterport3DLoader(
                Matterport3D_root=self.matterport3d_data_root,
                img_size=self.image_size,
                stage="test"
            )
        if stage == "test":
            self.test_dataset = Matterport3DLoader(
                Matterport3D_root=self.matterport3d_data_root,
                img_size=self.image_size,
                stage="test"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)


class ToFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tof18_data_root: str = "",
        img_size: Tuple = (480, 640),
        batch_size: int = 32,
        num_workers: int = 16,
        use_labels: bool = True,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        # Set the seed for PyTorch
        torch.manual_seed(42)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_labels = use_labels
        self.num_cluster = num_cluster

        dataset = ToFLoader(
            tof18_data_root,
            img_size=img_size,
        )
        dataset_train, dataset_valid = random_split(dataset, [0.8, 0.2])

        self.train_dataset = dataset_train
        self.valid_dataset = dataset_valid
        self.test_dataset = dataset_valid

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)


class DIODEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        DIODE_data_root: str = "",
        img_size: Tuple = (192, 256),
        batch_size: int = 32,
        num_workers: int = 16,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        # Set the seed for PyTorch
        # torch.manual_seed(42)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_cluster = num_cluster

        self.DIODE_data_root = DIODE_data_root
        self.image_size = img_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = DIODELoader(
                DIODE_data_root=self.DIODE_data_root,
                img_size=self.image_size,
                stage="train"
            )
            self.valid_dataset = DIODELoader(
                DIODE_data_root=self.DIODE_data_root,
                img_size=self.image_size,
                stage="val"
            )
        if stage == "test":
            self.test_dataset = DIODELoader(
                DIODE_data_root=self.DIODE_data_root,
                img_size=self.image_size,
                stage="val"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)
