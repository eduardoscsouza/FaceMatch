from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import os



def get_bboxs_generator(bboxs_df, imgs_dir="../data/Img_Resize", batch_size=32,
                    out_image_size=(56, 56), color='rgb', preprocess_func=None,
                    shuffle=True, seed=None,
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
                        rescale=1.0/255.0,
                        preprocessing_function=preprocess_func,
                        data_format='channels_last',
                        validation_split=0.0,
                        dtype=None)

    if flow_args is None:
        flow_args = dict(directory=imgs_dir,
                        x_col="image_id",
                        y_col=list(bboxs_df.columns[1:]),
                        weight_col=None,
                        target_size=out_image_size,
                        color_mode=color,
                        classes=None,
                        class_mode='raw',
                        batch_size=batch_size,
                        shuffle=shuffle,
                        seed=seed,
                        save_to_dir=None,
                        save_prefix='',
                        save_format='jpg',
                        subset=None,
                        interpolation='bilinear',
                        validate_filenames=True)

    return ImageDataGenerator(**gen_args).flow_from_dataframe(bboxs_df, **flow_args)



class FaceTripleGenerator(Sequence):
    def __init__(self, indvs_df, min_indv_imgs=5, imgs_dir="../data/Img_Crop_Resize",
                batch_size=32, out_dtype=np.float32, out_color='rgb',
                resize=False, cv2_resize_inter=cv2.INTER_LINEAR, out_image_size=(224, 224),
                preprocess_func=None):
        self.indvs = indvs_df.groupby(indvs_df.columns[1])
        self.indvs = [np.asarray(imgs.iloc[:, 0]) for _, imgs in self.indvs if len(imgs) >= min_indv_imgs]
        self.__aux_len__ = len(self.indvs)
        self.__aux_indvs_len__ = [len(indv) for indv in self.indvs]

        self.imgs_dir = imgs_dir
        self.batch_size = batch_size
        self.out_dtype = out_dtype
        self.out_color = cv2.COLOR_BGR2GRAY if (out_color == 'gray') else cv2.COLOR_BGR2RGB
        self.resize = resize
        self.cv2_resize_inter = cv2_resize_inter
        self.out_image_size = out_image_size
        self.preprocess_func = preprocess_func

    def __len__(self):
        return self.__aux_len__

    def __get_random_triple__(self):
        pos_indv, neg_indv = np.random.choice(self.__aux_len__, size=(2,), replace=False)
        anchor, pos = np.random.choice(self.__aux_indvs_len__[pos_indv], size=(2,), replace=False)
        neg = np.random.randint(self.__aux_indvs_len__[neg_indv])

        imgs = [self.indvs[pos_indv][anchor], self.indvs[pos_indv][pos], self.indvs[neg_indv][neg]]
        imgs = [cv2.imread(os.path.join(self.imgs_dir, img), cv2.IMREAD_COLOR) for img in imgs]
        imgs = imgs if not self.resize else [cv2.resize(img, self.out_image_size, interpolation=self.cv2_resize_inter) for img in imgs]
        imgs = [cv2.cvtColor(img, self.out_color) for img in imgs]

        return np.stack(imgs, axis=0)

    def __getitem__(self, _):
        indvs = [self.__get_random_triple__() for _ in range(self.batch_size)]
        indvs = np.stack(indvs, axis=0).astype(self.out_dtype)
        indvs = [indvs[:, 0], indvs[:, 1], indvs[:, 2]]
        indvs = indvs if self.preprocess_func is None else [self.preprocess_func(indv) for indv in indvs]

        return indvs

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv("../data/indvs.csv")
    gen = FaceTripleGenerator(df, imgs_dir="../data/Img_Crop_Resize")
    it = gen.__getitem__(0)
    for i in range(32):
        d = os.makedirs("temp/{}".format(i), exist_ok=True)
        cv2.imwrite("temp/{}/{}.jpg".format(i, "0"), cv2.cvtColor(it[0][i], cv2.COLOR_RGB2BGR))
        cv2.imwrite("temp/{}/{}.jpg".format(i, "1"), cv2.cvtColor(it[1][i], cv2.COLOR_RGB2BGR))
        cv2.imwrite("temp/{}/{}.jpg".format(i, "2"), cv2.cvtColor(it[2][i], cv2.COLOR_RGB2BGR))