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

class TripletTrainGenerator(Sequence):
    def __init__(self, indvs_df, min_indv_imgs=5, imgs_dir="../data/Img_Crop_Resize",
                batch_n_indvs=4, batch_indv_n_imgs=4,
                out_dtype=np.float32, out_color='rgb',
                resize=False, out_image_size=(224, 224), cv2_inter=cv2.INTER_LINEAR,
                preprocess_func=None):
        self.indvs = indvs_df.groupby(indvs_df.columns[1])
        self.indvs = [np.asarray(imgs.iloc[:, 0]) for _, imgs in self.indvs if len(imgs) >= min_indv_imgs]
        self.__aux_len__ = len(self.indvs)
        self.__aux_indvs_len__ = [len(indv) for indv in self.indvs]
        self.__len_out__ = np.sum(self.__aux_indvs_len__)

        cv2_color_BGR2 = cv2.COLOR_BGR2GRAY if (out_color == 'gray') else cv2.COLOR_BGR2RGB
        self.__load_img_args__ = (resize, out_image_size, cv2_inter, cv2_color_BGR2)

        self.imgs_dir = imgs_dir
        self.batch_n_indvs = batch_n_indvs
        self.batch_indv_n_imgs = batch_indv_n_imgs
        self.out_dtype = out_dtype
        self.preprocess_func = preprocess_func

    def __len__(self):
        return self.__len_out__

    def __getitem__(self, _):
        batch_indvs = np.random.choice(self.__aux_len__, size=(self.batch_n_indvs,), replace=False)

        batch_indvs_imgs = [np.random.choice(self.__aux_indvs_len__[indv], size=(self.batch_indv_n_imgs,), replace=False) for indv in batch_indvs]
        batch_indvs_imgs = [[__load_img__(os.path.join(self.imgs_dir, self.indvs[indv][img]), *self.__load_img_args__)
                            for img in indv_imgs] for indv, indv_imgs in zip(batch_indvs, batch_indvs_imgs)]
        batch_indvs_imgs = np.concatenate([np.stack(indv_imgs, axis=0) for indv_imgs in batch_indvs_imgs], axis=0).astype(self.out_dtype)
        batch_indvs_imgs = batch_indvs_imgs if self.preprocess_func is None else self.preprocess_func(batch_indvs_imgs)

        batch_labels = [np.asarray([label]*self.batch_indv_n_imgs) for label in batch_indvs]
        batch_labels = np.concatenate(batch_labels, axis=0).astype(np.int32)

        return batch_indvs_imgs, batch_labels

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv("../data/indvs.csv")
    gen = TripletTrainGenerator(df, imgs_dir="../data/Img_Crop_Resize")
    out_dir = os.makedirs("temp", exist_ok=True)
    imgs, labels = gen.__getitem__(0)
    print(labels)
    for i in range(16):
        cv2.imwrite("temp/{}.jpg".format(i), cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))