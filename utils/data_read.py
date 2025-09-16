import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import torch.utils
import torch.utils.data


def pca_whitening(image, number_of_pc):

    shape = image.shape

    image = np.reshape(image, [shape[0]*shape[1], shape[2]])
    number_of_rows = shape[0]
    number_of_columns = shape[1]
    pca = PCA(n_components = number_of_pc)
    image = pca.fit_transform(image)
    pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc),dtype=np.float32)
    for i in range(number_of_pc):
        pc_images[:, :, i] = np.reshape(image[:, i], (number_of_rows, number_of_columns))
    return pc_images


def load_data(dataset):
    ###下面是讲解python怎么读取.mat文件以及怎么处理得到的结果###
    data_dir = './datasets'
    if dataset == 'IP':
        image_file = data_dir + '/IP/Indian_pines_corrected.mat'
        label_file = data_dir + '/IP/Indian_pines_gt.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
        image = image_data['indian_pines_corrected']
        label = label_data['indian_pines_gt']
        label_values = [
            "0.Undefined",
            "1.Alfalfa",
            "2.Corn-notill",
            "3.Corn-mintill",
            "4.Corn",
            "5.Grass-pasture",
            "6.Grass-trees",
            "7.Grass-pasture-mowed",
            "8.Hay-windrowed",
            "9.Oats",
            "10.Soybean-notill",
            "11.Soybean-mintill",
            "12.Soybean-clean",
            "13.Wheat",
            "14.Woods",
            "15.Buildings-Grass-Trees-Drives",
            "16.Stone-Steel-Towers",
        ]
        palette = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Magenta
            6: (0, 255, 255),  # Cyan
            7: (128, 0, 0),  # Dark Red
            8: (0, 128, 0),  # Dark Green
            9: (0, 0, 128),  # Dark Blue
            10: (128, 128, 0),  # Olive
            11: (128, 0, 128),  # Purple
            12: (0, 128, 128),  # Teal
            13: (192, 192, 192),  # Light Grey
            14: (64, 64, 64),  # Dark Grey
            15: (255, 128, 0),  # Orange
            16: (128, 128, 255)  # Light Blue
        }
    elif dataset == 'PU':
        image_file = data_dir + '/PU/PaviaU.mat'
        label_file = data_dir + '/PU/PaviaU_gt.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
        image = image_data['paviaU']#pavia1
        label = label_data['paviaU_gt']#pavia1
        label_values = [
            "0.Undefined",
            "1.Asphalt",
            "2.Meadows",
            "3.Gravel",
            "4.Trees",
            "5.Painted metal sheets",
            "6.Bare Soil",
            "7.Bitumen",
            "8.Self-Blocking Bricks",
            "9.Shadows",
        ]
        palette = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Magenta
            6: (0, 255, 255),  # Cyan
            7: (128, 0, 0),  # Dark Red
            8: (0, 128, 0),  # Dark Green
            9: (0, 0, 128),  # Dark Blue
        }
    elif dataset == 'HU2013':
        image_file = data_dir + '/HU2013/Houston.mat'
        label_file = data_dir + '/HU2013/Houston_gt.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
        image = image_data['Houston']
        label = label_data['Houston_gt']  # houston
        label_values = ['0.Undefined', '1.Healthy grass', '2.Stressed grass', '3.Synthetic grass', '4.Trees',
                        '5.Soil', '6.Water', '7.Residential', '8.Commercial', '9.Road', '10.Highway',
                        '11.Railway', '12.Parking Lot1', '13.Parking Lot2', '14.Tennis court', '15.Running track']
        palette = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Magenta
            6: (0, 255, 255),  # Cyan
            7: (128, 0, 0),  # Dark Red
            8: (0, 128, 0),  # Dark Green
            9: (0, 0, 128),  # Dark Blue
            10: (128, 128, 0),  # Olive
            11: (128, 0, 128),  # Purple
            12: (0, 128, 128),  # Teal
            13: (192, 192, 192),  # Light Grey
            14: (64, 64, 64),  # Dark Grey
            15: (255, 128, 0),  # Orange
        }
    elif dataset == 'WHU-HC':
        image_file = data_dir + '/WHU-HanChuan/WHU_Hi_HanChuan.mat'
        label_file = data_dir + '/WHU-HanChuan/WHU_Hi_HanChuan_gt.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
        image = image_data['WHU_Hi_HanChuan']
        label = label_data['WHU_Hi_HanChuan_gt']  # houston
        label_values = [
            '0.Undefined', 
            '1.Strawberry', 
            '2.Cowpea', 
            '3.Soybean', 
            '4.Sorghum',
            '5.Water spinach', 
            '6.Watermelon', 
            '7.Greens', 
            '8.Trees', 
            '9.Grass', 
            '10.Red roof',
            '11.Gray roof', 
            '12.Plastic', 
            '13.Bare soil', 
            '14.Road', 
            '15.Bright object',
            '16.Water']
        palette = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Magenta
            6: (0, 255, 255),  # Cyan
            7: (128, 0, 0),  # Dark Red
            8: (0, 128, 0),  # Dark Green
            9: (0, 0, 128),  # Dark Blue
            10: (128, 128, 0),  # Olive
            11: (128, 0, 128),  # Purple
            12: (0, 128, 128),  # Teal
            13: (192, 192, 192),  # Light Grey
            14: (64, 64, 64),  # Dark Grey
            15: (255, 128, 0),  # Orange
            16: (128, 128, 255),  # Light Blue
        }
    elif dataset == 'WHU-HH':
        image_file = data_dir + '/WHU-HongHu/WHU_Hi_HongHu.mat'
        label_file = data_dir + '/WHU-HongHu/WHU_Hi_HongHu_gt.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
        image = image_data['WHU_Hi_HongHu']
        label = label_data['WHU_Hi_HongHu_gt'] 
        label_values = [
            '0.Undefined', 
            '1.Red roof', 
            '2.Road', 
            '3.Bare soil', 
            '4.Cotton',
            '5.Cotton firewood', 
            '6.Rape', 
            '7.Chinese cabbage', 
            '8.Pakchoi', 
            '9.Cabbage', 
            '10.Tuber mustard',
            '11.Brassica parachinensis', 
            '12.Brassica chinensis', 
            '13.Small Brassica chinensis', 
            '14.Lactuca sativa', 
            '15.Celtuce',
            '16.Film covered lettuce',
            '17.Romaine lettuce',
            '18.Carrot',
            '19.White radish',
            '20.Garlic sprout',
            '21.Broad bean',
            '22.Tree']
        palette = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Magenta
            6: (0, 255, 255),  # Cyan
            7: (128, 0, 0),  # Dark Red
            8: (0, 128, 0),  # Dark Green
            9: (0, 0, 128),  # Dark Blue
            10: (128, 128, 0),  # Olive
            11: (128, 0, 128),  # Purple
            12: (0, 128, 128),  # Teal
            13: (192, 192, 192),  # Light Grey
            14: (64, 64, 64),  # Dark Grey
            15: (255, 128, 0),  # Orange
            16: (128, 128, 255),  # Light Blue
            17: (255, 128, 255), #
            18: (0, 255, 128),
            19: (255, 128, 128),
            20: (128, 255, 128),
            21: (255, 0, 128),
            22: (128, 0, 255),
        }

    else:
        raise Exception('dataset does not find')

    # Filter NaN out
    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN "
            "data is disabled."
        )
    image[nan_mask] = 0
    label[nan_mask] = 0
    
    image = image.astype(np.float32)

    return image, label, palette, label_values


def readdata(windowsize, args):

    or_image, or_label, palette, label_values = load_data(args.dataset)
    windowsize = windowsize
    train_num = args.train_num
    halfsize = int((windowsize-1)/2)
    number_class = np.max(or_label)
    
    #preprocessing
    image = np.pad(or_image, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
    label = np.pad(or_label, ((halfsize, halfsize), (halfsize, halfsize)), 'constant',constant_values=0)
    if args.type == 'PCA':
        image1 = pca_whitening(image, number_of_pc = 30)
    if args.type == 'none':
        image1 = image
    image = image1
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    return image, label, palette, label_values


def sample_gt(gt, train_size, num, mode='fixed'):
    """
    Extract a fixed number of samples or a percentage of samples from an array of labels for training and testing.

    Args:
        gt: a 2D array of int labels
        train_size: an int number of samples or a float percentage [0, 1] of samples to use for training
        mode: a string specifying the sampling strategy, options include 'random', 'fixed', 'disjoint'

    Returns:
        train_gt: a 2D array of int labels for training
        test_gt: a 2D array of int labels for testing
    """
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # (x,y）
    y = gt[indices].ravel()  # label
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)

    if train_size > 1:
        train_size = int(train_size)
    if isinstance(train_size, float):
        train_size = int(train_size * y.size)

    if mode == 'random':
        train_indices, test_indices = train_test_split(X, train_size=train_size, stratify=y if train_size > 1 else None)
        train_gt[tuple(zip(*train_indices))] = gt[tuple(zip(*train_indices))]
        test_gt[tuple(zip(*test_indices))] = gt[tuple(zip(*test_indices))]
    elif mode == 'fixed':
        for c in np.unique(gt):
            if c == 0:
                continue
            count_c = np.sum(gt == c)
            if count_c <= train_size:
                split_size = 15
            else:
                split_size = train_size
            if count_c == 15:
                split_size = 5
            indices = np.nonzero(gt == c)
            X = np.array(list(zip(*indices))) 
            train_indices, test_indices = train_test_split(X, train_size=split_size, random_state=42+num, stratify=None)
            train_gt[tuple(zip(*train_indices))] = gt[tuple(zip(*train_indices))]
            test_gt[tuple(zip(*test_indices))] = gt[tuple(zip(*test_indices))]
    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError(f"{mode} sampling is not implemented yet.")

    return train_gt, test_gt


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, args):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = args.dataset
        self.patch_size = args.windowsize
        self.ignored_labels = args.ignored_labels
        self.flip_augmentation = args.flip_aug
        self.radiation_augmentation = args.radiation_aug
        self.mixture_augmentation = args.mixture_aug
        self.center_pixel = args.center_pixel
        self.supervision = args.supervision

        # Fully supervised : use all pixels with label not ignored
        if self.supervision == "full":
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif self.supervision == "semi":
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        # x_pos, y_pos = np.nonzero(gt)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        

        return data, label
    
def build_loader(dataset_train, dataset_val, dataset_test, batch_size):

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True) if dataset_train is not None else None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False) if dataset_val is not None else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False) if dataset_test is not None else None
    
    return data_loader_train, data_loader_val, data_loader_test


