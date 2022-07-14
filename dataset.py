import os.path as osp
import os
from utils.data_augumentation import (
    Compose,
    Scale,
    RandomRotation,
    RandomMirror,
    Resize,
    Normalize_Tensor,
)
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
from statistics import mean, stdev
from sklearn.model_selection import train_test_split


def select_file_with_json(rootpath):
    """
    jsonファイルを参照してbutuと背景クラスのみが入っているファイルのみのファイル名をリスト化する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : file_names
        データへのパスを格納したリスト
    """
    json_dir = osp.join(rootpath, "json")
    json_tempate = osp.join(rootpath, "json")
    # print(json_dir)
    # print(osp.exists(json_dir))
    json_files = os.listdir(json_dir)
    filenames = []
    for json_file in json_files:
        path = osp.join(json_tempate, json_file)
        json_open = open(path, "r")
        json_load = json.load(json_open)
        shapes = json_load["shapes"]
        for shape in shapes:
            if shape["label"] != "butu":
                is_butu = False
                break
            is_butu = True
        if is_butu:
            filenames.append(json_file)
    # print(len(filenames))
    return filenames


def make_datapath_list(rootpath, filenames):
    """
    学習、検証の画像データとアノテーションデータのファイルパスリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """
    image_files = [
        osp.join(rootpath, "image", "%s.png" % os.path.splitext(filename)[0])
        for filename in filenames
    ]
    train_img_list = list()
    train_anno_list = list()
    val_img_list = list()
    val_anno_list = list()
    test_img_list = list()
    test_anno_list = list()
    annotate_files = [
        osp.join(
            rootpath, "SegmentationClassPNG", "%s.png" % os.path.splitext(filename)[0]
        )
        for filename in filenames
    ]
    # print(len(image_files))
    # print(image_files[0])
    # print(len(annotate_files))
    # print(annotate_files[0])
    num_train = len(image_files) * 8 // 10
    num_val = (len(image_files) - num_train) // 2
    num_test = len(image_files) - num_train - num_val
    # print(num_train, num_val, num_test)

    index = 0
    for image, anno in zip(image_files, annotate_files):
        if index < num_train:
            train_img_list.append(image)
            train_anno_list.append(anno)
        elif index < num_train + num_val:
            val_img_list.append(image)
            val_anno_list.append(anno)
        else:
            test_img_list.append(image)
            test_anno_list.append(anno)
        index += 1

    return (
        train_img_list,
        train_anno_list,
        val_img_list,
        val_anno_list,
        test_img_list,
        test_anno_list,
    )


def make_random_datapath_list(rootpath, filenames):
    """
    学習、検証の画像データとアノテーションデータのファイルパスリストをランダムに作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """
    img_path_list = [
        osp.join(rootpath, "image", "%s.png" % os.path.splitext(filename)[0])
        for filename in filenames
    ]
    anno_path_list = [
        osp.join(
            rootpath, "SegmentationClassPNG", "%s.png" % os.path.splitext(filename)[0]
        )
        for filename in filenames
    ]
    num_train = len(img_path_list) * 8 // 10
    num_val = (len(img_path_list) - num_train) // 2
    num_test = len(img_path_list) - num_train - num_val
    (tmp_img, test_img_list, tmp_anno, test_anno_list) = train_test_split(
        img_path_list,
        anno_path_list,
        train_size=num_train + num_val,
        test_size=num_test,
        shuffle=True,
        random_state=1,
    )

    (train_img_list, val_img_list, train_anno_list, val_anno_list) = train_test_split(
        tmp_img,
        tmp_anno,
        train_size=num_train,
        test_size=num_val,
        shuffle=True,
        random_state=1,
    )
    return (
        train_img_list,
        train_anno_list,
        val_img_list,
        val_anno_list,
        test_img_list,
        test_anno_list,
    )


class DataTransform:
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": Compose(
                [
                    Scale(scale=[0.5, 1.5]),
                    RandomRotation(angle=[-10, 10]),
                    RandomMirror(),
                    Resize(input_size),
                    Normalize_Tensor(color_mean, color_std),
                ]
            ),
            "val": Compose(
                [
                    Resize(input_size),
                    Normalize_Tensor(color_mean, color_std),
                ]
            ),
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'val'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        """画像の枚数を返す"""
        return len(self.img_list)

    def __getitem__(self, index):
        """
        前処理をした画像のTensor形式のデータとアノテーションを取得
        """
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        """画像のTensor形式のデータ、アノテーションを取得する"""

        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)  # [高さ][幅]
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)  # [高さ][幅]
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img


def make_color_mean_std(train_img_list, val_img_list, test_img_list):
    all_list = np.concatenate([train_img_list, val_img_list, test_img_list])
    mean_list = []
    std_list = []
    for path in all_list:
        img = Image.open(path)
        img = np.array(img)
        mean_list.append(np.mean(img))
        std_list.append(np.std(img))
    return np.mean(mean_list) / 255, np.mean(std_list) / 255


##以下はテスト用関数


def make_datapath_list_test():
    rootpath = "../data/x3"
    filenames = select_file_with_json(rootpath)
    (
        train_img_list,
        train_anno_list,
        val_img_list,
        val_anno_list,
        test_img_list,
        test_anno_list,
    ) = make_datapath_list(rootpath, filenames)
    print(test_img_list)


def test_make_random_datapath_list():
    rootpath = "../data/x3"
    rootpath = "../data/x3"
    filenames = select_file_with_json(rootpath)
    make_random_datapath_list(rootpath, filenames)


def make_dataset_test():
    rootpath = "../data/x3"
    filenames = select_file_with_json(rootpath)
    # print(filenames)
    (
        train_img_list,
        train_anno_list,
        val_img_list,
        val_anno_list,
        test_img_list,
        test_anno_list,
    ) = make_datapath_list(rootpath, filenames)
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    val_dataset = VOCDataset(
        val_img_list,
        val_anno_list,
        phase="val",
        transform=DataTransform(
            input_size=475, color_mean=color_mean, color_std=color_std
        ),
    )

    # データの取り出し例
    print(val_dataset.__getitem__(0)[0].shape)
    print("======================")
    print(val_dataset.__getitem__(0)[1].shape)
    print("======================")
    print(val_dataset.__getitem__(0))


def show_dataset_test(index=0):
    rootpath = "../data/x3"
    filenames = select_file_with_json(rootpath)
    (
        train_img_list,
        train_anno_list,
        val_img_list,
        val_anno_list,
        test_img_list,
        test_anno_list,
    ) = make_datapath_list(rootpath, filenames)
    color_mean = 0.18228737997050898
    color_std = 0.15940997135888293
    train_dataset = VOCDataset(
        train_img_list,
        train_anno_list,
        phase="train",
        transform=DataTransform(
            input_size=475, color_mean=color_mean, color_std=color_std
        ),
    )
    imges, anno_class_imges = train_dataset.__getitem__(index)
    img_val = imges
    print(img_val[0].shape)
    img_val = img_val[0].detach().numpy()
    plt.imshow(img_val, cmap="gray")
    plt.savefig("../class3/memo_img.png")

    # アノテーション画像の表示
    anno_file_path = train_anno_list[index]
    anno_class_img = Image.open(anno_file_path)  # [高さ][幅][色RGB]
    p_palette = anno_class_img.getpalette()

    anno_class_img_val = anno_class_imges.numpy()
    anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
    anno_class_img_val.putpalette(p_palette)
    plt.imshow(anno_class_img_val)
    plt.savefig("../class3/memo_anno.png")
    print(train_img_list[index], train_anno_list[index])


def select_file_with_json_test():
    rootpath = "../data/x3"
    select_file_with_json(rootpath)


def test_make_color_mean_std():
    rootpath = "../data/x3"
    filenames = select_file_with_json(rootpath)
    (
        train_img_list,
        train_anno_list,
        val_img_list,
        val_anno_list,
        test_img_list,
        test_anno_list,
    ) = make_datapath_list(rootpath, filenames)
    color_mean, color_std = make_color_mean_std(
        train_img_list, val_img_list, test_img_list
    )
    print(color_mean, color_std)


# make_datapath_list_test()
# make_dataset_test()
# show_dataset_test(index=2)
# select_file_with_json_test()
# test_make_color_mean_std()
test_make_random_datapath_list()
