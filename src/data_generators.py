from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import os



def get_bboxs_generator(bboxs_df, imgs_dir="../data/Img_Resize", batch_size=32,
                    out_image_size=(56, 56), resize_inter='bilinear',
                    color_mode='rgb', rescale=1.0/255.0, preprocess_func=None,
                    gen_args=None, flow_args=None):

    if gen_args is None:
        gen_args = dict(featurewise_center=False,
                        samplewise_center=False,
                        featurewise_std_normalization=False,
                        samplewise_std_normalization=False,
                        zca_whitening=False,
                        zca_epsilon=1e-06,
                        rotation_range=0.0,
                        width_shift_range=0.0,
                        height_shift_range=0.0,
                        brightness_range=None,
                        shear_range=0.0,
                        zoom_range=0.0,
                        channel_shift_range=0.0,
                        fill_mode='nearest',
                        cval=0.0,
                        horizontal_flip=False,
                        vertical_flip=False,
                        rescale=rescale,
                        preprocessing_function=preprocess_func,
                        data_format='channels_last',
                        validation_split=0.0,
                        dtype=None)

    if flow_args is None:
        flow_args = dict(directory=imgs_dir,
                        x_col=bboxs_df.columns[0],
                        y_col=list(bboxs_df.columns[1:]),
                        weight_col=None,
                        target_size=out_image_size,
                        color_mode=color_mode,
                        classes=None,
                        class_mode='raw',
                        batch_size=batch_size,
                        shuffle=True,
                        seed=None,
                        save_to_dir=None,
                        save_prefix='',
                        save_format='jpg',
                        subset=None,
                        interpolation=resize_inter,
                        validate_filenames=True)

    return ImageDataGenerator(**gen_args).flow_from_dataframe(bboxs_df, **flow_args)



def __load_img__(img_path, resize, out_image_size, cv2_inter, cv2_color_BGR2):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = img if not resize else cv2.resize(img, out_image_size, interpolation=cv2_inter)
    img = cv2.cvtColor(img, cv2_color_BGR2)

    return img

def __get_indvs__(indvs_df, min_indv_imgs, imgs_dir):
    indvs = indvs_df.groupby(indvs_df.columns[1])
    indvs = [imgs.iloc[:, 0] for _, imgs in indvs if len(imgs) >= min_indv_imgs]
    indvs = [np.asarray([os.path.join(imgs_dir, img) for img in imgs]) for imgs in indvs]

    aux_len = len(indvs)
    aux_indvs_len = [len(indv) for indv in indvs]
    len_out = np.sum(aux_indvs_len)

    return indvs, aux_len, aux_indvs_len, len_out

def __get_load_img_args__(out_color, resize, out_image_size, cv2_inter):
    cv2_color_BGR2 = cv2.COLOR_BGR2GRAY if (out_color == 'gray') else cv2.COLOR_BGR2RGB
    return (resize, out_image_size, cv2_inter, cv2_color_BGR2)

def normalize_255(imgs):
    return imgs / 255.0



class BBoxsGenerator(Sequence):
    def __init__(self, bboxs_df, imgs_dir="../data/Img_Resize",
                batch_size=32,
                out_dtype=np.float32, out_color='rgb',
                resize=False, out_image_size=(56, 56), cv2_inter=cv2.INTER_LINEAR,
                preprocess_func=normalize_255, preprocess_include_label=False):
        self.imgs = np.asarray([os.path.join(imgs_dir, img) for img in bboxs_df.iloc[:, 0]])
        self.labels = np.asarray(bboxs_df.iloc[:, 1:], dtype=np.float64 if (out_dtype==np.float64) else np.float32)
        self.__aux_len__ = len(self.imgs)
        self.__len_out__ = len(self.imgs) // batch_size

        self.ord = np.random.permutation(self.__aux_len__)
        self.__load_img_args__ = __get_load_img_args__(out_color, resize, out_image_size, cv2_inter)

        self.batch_size = batch_size
        self.out_dtype = out_dtype
        self.preprocess_func = preprocess_func
        self.preprocess_include_label = preprocess_include_label

    def __len__(self):
        return self.__len_out__

    def __getitem__(self, index):
        cut = self.ord[self.batch_size*index:self.batch_size*(index+1)]

        batch_imgs = [__load_img__(img, *self.__load_img_args__) for img in self.imgs[cut]]
        batch_imgs = np.stack(batch_imgs, axis=0).astype(self.out_dtype)
        batch_labels = self.labels[cut]

        if self.preprocess_include_label:
            batch_imgs, batch_labels = (batch_imgs, batch_labels) if self.preprocess_func is None else self.preprocess_func(batch_imgs, batch_labels)
        else:
            batch_imgs = batch_imgs if self.preprocess_func is None else self.preprocess_func(batch_imgs)

        return batch_imgs, batch_labels



class TripletTrainGenerator(Sequence):
    def __init__(self, indvs_df, min_indv_imgs=5, imgs_dir="../data/Img_Crop_Resize",
                batch_n_indvs=4, batch_indv_n_imgs=4,
                out_dtype=np.float32, out_color='rgb',
                resize=False, out_image_size=(224, 224), cv2_inter=cv2.INTER_LINEAR,
                preprocess_func=None):
        self.indvs, self.__aux_len__, self.__aux_indvs_len__, self.__len_out__ = __get_indvs__(indvs_df, min_indv_imgs, imgs_dir)
        self.__load_img_args__ = __get_load_img_args__(out_color, resize, out_image_size, cv2_inter)

        self.batch_n_indvs = batch_n_indvs
        self.batch_indv_n_imgs = batch_indv_n_imgs
        self.out_dtype = out_dtype
        self.preprocess_func = preprocess_func

    def __len__(self):
        return self.__len_out__

    def __getitem__(self, _):
        batch_indvs = np.random.choice(self.__aux_len__, size=(self.batch_n_indvs,), replace=False)

        batch_indvs_imgs = [np.random.choice(self.__aux_indvs_len__[indv], size=(self.batch_indv_n_imgs,), replace=False) for indv in batch_indvs]
        batch_indvs_imgs = [[__load_img__(self.indvs[indv][img], *self.__load_img_args__) for img in indv_imgs]
                            for indv, indv_imgs in zip(batch_indvs, batch_indvs_imgs)]
        batch_indvs_imgs = np.concatenate([np.stack(indv_imgs, axis=0) for indv_imgs in batch_indvs_imgs], axis=0).astype(self.out_dtype)
        batch_indvs_imgs = batch_indvs_imgs if self.preprocess_func is None else self.preprocess_func(batch_indvs_imgs)

        batch_labels = [np.asarray([label]*self.batch_indv_n_imgs) for label in batch_indvs]
        batch_labels = np.concatenate(batch_labels, axis=0).astype(np.int32)

        return batch_indvs_imgs, batch_labels



class TripletDistancesGenerator(Sequence):
    def __init__(self, indvs_df, min_indv_imgs=5, imgs_dir="../data/Img_Crop_Resize",
                batch_size=32,
                out_dtype=np.float32, out_color='rgb',
                resize=False, out_image_size=(224, 224), cv2_inter=cv2.INTER_LINEAR,
                preprocess_func=None):
        self.indvs, self.__aux_len__, self.__aux_indvs_len__, self.__len_out__ = __get_indvs__(indvs_df, min_indv_imgs, imgs_dir)
        self.__load_img_args__ = __get_load_img_args__(out_color, resize, out_image_size, cv2_inter)

        self.batch_size = batch_size
        self.out_dtype = out_dtype
        self.preprocess_func = preprocess_func

    def __len__(self):
        return self.__len_out__

    def __get_random_triple__(self):
        pos_indv, neg_indv = np.random.choice(self.__aux_len__, size=(2,), replace=False)
        anchor, pos = np.random.choice(self.__aux_indvs_len__[pos_indv], size=(2,), replace=False)
        neg = np.random.randint(self.__aux_indvs_len__[neg_indv])

        imgs = [self.indvs[pos_indv][anchor], self.indvs[pos_indv][pos], self.indvs[neg_indv][neg]]
        imgs = [__load_img__(img, *self.__load_img_args__) for img in imgs]

        return np.stack(imgs, axis=0)

    def __getitem__(self, _):
        indvs = [self.__get_random_triple__() for _ in range(self.batch_size)]
        indvs = np.stack(indvs, axis=0).astype(self.out_dtype)
        indvs = [indvs[:, 0], indvs[:, 1], indvs[:, 2]]
        indvs = indvs if self.preprocess_func is None else [self.preprocess_func(indv) for indv in indvs]

        return indvs



class TripletClassifierGenerator(Sequence):
    def __init__(self, indvs_df, min_indv_imgs=5, imgs_dir="../data/Img_Crop_Resize",
                batch_size=32,
                out_dtype=np.float32, out_color='rgb',
                resize=False, out_image_size=(224, 224), cv2_inter=cv2.INTER_LINEAR,
                preprocess_func=None):
        self.indvs, self.__aux_len__, self.__aux_indvs_len__, self.__len_out__ = __get_indvs__(indvs_df, min_indv_imgs, imgs_dir)
        self.__load_img_args__ = __get_load_img_args__(out_color, resize, out_image_size, cv2_inter)

        self.batch_size = batch_size
        self.out_dtype = out_dtype
        self.preprocess_func = preprocess_func

    def __len__(self):
        return self.__len_out__

    def __getitem__(self, _):
        labels = np.random.randint(2, size=self.batch_size)

        indvs = np.random.randint(self.__aux_len__, size=self.batch_size)
        indvs = [[indv, indv] if label else np.random.choice(self.__aux_len__, size=(2,), replace=False) for indv, label in zip(indvs, labels)]

        imgs = [np.random.choice(self.__aux_indvs_len__[indv[0]], size=(2,), replace=False) if label else
            [np.random.randint(self.__aux_indvs_len__[indv[0]]), np.random.randint(self.__aux_indvs_len__[indv[1]])]
            for indv, label in zip(indvs, labels)]
        imgs = [[__load_img__(self.indvs[indv[0]][img[0]], *self.__load_img_args__),
                __load_img__(self.indvs[indv[1]][img[1]], *self.__load_img_args__)]
                for indv, img in zip(indvs, imgs)]

        indvs = np.stack([np.stack(img, axis=0) for img in imgs], axis=0).astype(self.out_dtype)
        indvs = [indvs[:, 0], indvs[:, 1]]
        indvs = indvs if self.preprocess_func is None else [self.preprocess_func(indv) for indv in indvs]

        return indvs, labels



if __name__ == '__main__':
    import shutil
    import pandas as pd
    df = pd.read_csv("../data/indvs.csv")
    bboxs_df = pd.read_csv("../data/bboxs_x2y2.csv")
    imgs_dir = "../data/Img_Crop_Resize"

    out_dir = "temp_1"
    os.makedirs(out_dir, exist_ok=True)
    gen = TripletTrainGenerator(df, imgs_dir=imgs_dir)
    print(gen.__len__())
    imgs, labels = gen.__getitem__(0)
    print(labels)
    for i in range(16):
        cv2.imwrite("{}/{}.jpg".format(out_dir, i), cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))

    out_dir = "temp_2"
    os.makedirs(out_dir, exist_ok=True)
    gen = TripletDistancesGenerator(df, imgs_dir=imgs_dir)
    print(gen.__len__())
    imgs = gen.__getitem__(0)
    for i in range(32):
        for j in range(3):
            cv2.imwrite("{}/{}-{}.jpg".format(out_dir, i, j), cv2.cvtColor(imgs[j][i], cv2.COLOR_RGB2BGR))

    out_dir = "temp_3"
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    gen = TripletClassifierGenerator(df, imgs_dir=imgs_dir)
    print(gen.__len__())
    imgs, labels = gen.__getitem__(0)
    print(len(imgs), imgs[0].shape, imgs[1].shape)
    for i in range(32):
        for j in range(2):
            cv2.imwrite("{}/{}-{}-{}.jpg".format(out_dir, i, j, labels[i]), cv2.cvtColor(imgs[j][i], cv2.COLOR_RGB2BGR))

    out_dir = "temp_4"
    os.makedirs(out_dir, exist_ok=True)
    for img_size in [112, 224]:
        batch_size = 32
        gen_a = get_bboxs_generator(bboxs_df, out_image_size=(img_size, img_size), batch_size=batch_size)
        gen_b = BBoxsGenerator(bboxs_df, out_image_size=(img_size, img_size), resize=(img_size!=224), batch_size=batch_size)
        last_a, last_b = gen_a.__len__(), gen_b.__len__()

        assert (last_a-last_b == 0) or (last_a-last_b == 1)
        assert gen_a.__getitem__(last_b-1)[0].shape == gen_b.__getitem__(last_b-1)[0].shape
        if last_a != last_b:
            assert gen_a.__getitem__(last_a-1)[0].shape != gen_b.__getitem__(last_b-1)[0].shape

        n_batchs, n_imgs_batch = 4, 4
        for i in range(n_batchs):
            batch, imgs = np.random.randint(last_b), np.random.randint(batch_size, size=(n_imgs_batch,))
            imgs_a, imgs_b = gen_a.__getitem__(batch), gen_b.__getitem__(batch)
            # Only works if shuffling is disabled
            # Images are the same, but BBoxsGenerator images are sharper
            #assert np.allclose(imgs_a[1], imgs_b[1])

            imgs_a, imgs_b = imgs_a[0][imgs], imgs_b[0][imgs]
            imgs_a, imgs_b = (255.0*imgs_a).astype(np.uint8), (255.0*imgs_b).astype(np.uint8)
            for j in range(n_imgs_batch):
                cv2.imwrite("{}/{}-{}-{}-{}.jpg".format(out_dir, img_size, i, j, "0"), cv2.cvtColor(imgs_a[j], cv2.COLOR_RGB2BGR))
                cv2.imwrite("{}/{}-{}-{}-{}.jpg".format(out_dir, img_size, i, j, "1"), cv2.cvtColor(imgs_b[j], cv2.COLOR_RGB2BGR))

    from time import time
    n_tests = 100

    gen = TripletTrainGenerator(df, imgs_dir=imgs_dir, batch_n_indvs=6, batch_indv_n_imgs=4)
    t0 = time()
    _ = [gen.__getitem__(0) for _ in range(n_tests)]
    print(time() - t0)

    gen = TripletDistancesGenerator(df, imgs_dir=imgs_dir, batch_size=8)
    t0 = time()
    _ = [gen.__getitem__(0) for _ in range(n_tests)]
    print(time() - t0)

    gen = TripletClassifierGenerator(df, imgs_dir=imgs_dir, batch_size=12)
    t0 = time()
    _ = [gen.__getitem__(0) for _ in range(n_tests)]
    print(time() - t0)

    gen = BBoxsGenerator(bboxs_df, imgs_dir=imgs_dir, batch_size=24)
    t0 = time()
    _ = [gen.__getitem__(0) for _ in range(n_tests)]
    print(time() - t0)