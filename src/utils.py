from multiprocessing import Pool
from glob import glob
import numpy as np
import pandas as pd
import os
import cv2



def __get_shape__(img_filename):
    return cv2.imread(img_filename, cv2.IMREAD_COLOR).shape

def gen_bboxs_csv(bboxs_txt, imgs_dir, bboxs_csv_outfile="bboxs.csv", return_df=False, n_procs=16):
    df = pd.read_csv(bboxs_txt, engine="python", sep='\ +', skiprows=1, index_col=0, dtype={0:str, 1:np.int32, 2:np.int32, 3:np.int32, 4:np.int32})

    with Pool(n_procs) as pool:
        imgs_shapes = pool.map(__get_shape__, [os.path.join(imgs_dir, img) for img in df.index])

    heights, widths = np.asarray([shape[0] for shape in imgs_shapes]), np.asarray([shape[1] for shape in imgs_shapes])
    df["y_1"], df["height"] = df["y_1"] / heights, df["height"] / heights
    df["x_1"], df["width"] = df["x_1"] / widths, df["width"] / widths

    df.to_csv(bboxs_csv_outfile)
    if return_df:
        return df

def gen_bboxs_csv_x2y2(bboxs_txt, imgs_dir, bboxs_csv_outfile="bboxs_x2y2.csv", return_df=False, n_procs=16):
    df = pd.read_csv(bboxs_txt, engine="python", sep='\ +', skiprows=1, index_col=0, dtype={0:str, 1:np.int32, 2:np.int32, 3:np.int32, 4:np.int32})

    with Pool(n_procs) as pool:
        imgs_shapes = pool.map(__get_shape__, [os.path.join(imgs_dir, img) for img in df.index])

    df["x_2"] = df["x_1"] + df["width"]
    df["y_2"] = df["y_1"] + df["height"]
    df = df.drop(columns=["width", "height"])

    heights, widths = np.asarray([shape[0] for shape in imgs_shapes]), np.asarray([shape[1] for shape in imgs_shapes])
    df["y_1"], df["y_2"] = df["y_1"] / heights, df["y_2"] / heights
    df["x_1"], df["x_2"] = df["x_1"] / widths, df["x_2"] / widths

    df.to_csv(bboxs_csv_outfile)
    if return_df:
        return df

def gen_splits_csv(splits_txt, splits_csv_outfile="splits.csv", return_df=False):
    df = pd.read_csv(splits_txt, sep=' ', index_col=0, names=["image_id", "split"], dtype={0:str, 1:np.int32})

    splits_names = {0:"train", 1:"val", 2:"test"}
    df["split"] = [splits_names[split] for split in df["split"]]

    df.to_csv(splits_csv_outfile)
    if return_df:
        return df

def gen_indvs_csv(indvs_txt, indvs_csv_outfile="indvs.csv", return_df=False):
    df = pd.read_csv(indvs_txt, sep=' ', index_col=0, names=["image_id", "indv_id"], dtype={0:str, 1:np.int32})

    df.to_csv(indvs_csv_outfile)
    if return_df:
        return df



def get_train_val_test_dfs(bboxs_csv, splits_csv):
    bboxs_df = pd.read_csv(bboxs_csv)
    splits_df = pd.read_csv(splits_csv)

    splits = splits_df["split"]
    train_df, val_df, test_df = bboxs_df[splits=="train"], bboxs_df[splits=="val"], bboxs_df[splits=="test"]

    return train_df, val_df, test_df



def __standardize_img__(img_filename, out_dir, out_size):
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
    img = cv2.resize(img, out_size, interpolation=cv2.INTER_CUBIC)

    out_name = '.'.join(img_filename.split('/')[-1].split('.')[:-1]) + ".jpg"
    cv2.imwrite(os.path.join(out_dir, out_name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

def standardize_imgs_files(imgs_dir, out_dir, out_size=(224, 224), n_procs=16):
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob(os.path.join(imgs_dir, "*.jpg")))
    with Pool(n_procs) as pool:
        _ = pool.starmap(__standardize_img__, [(img, out_dir, out_size) for img in imgs])



def __crop_standardize_img__(img_filename, crop_values, out_dir, out_size):
    x, y, width, height = crop_values
    x2, y2 = x+width, y+height

    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
    img = img[y:y2, x:x2]
    img = cv2.resize(img, out_size, interpolation=cv2.INTER_CUBIC)

    out_name = '.'.join(img_filename.split('/')[-1].split('.')[:-1]) + ".jpg"
    cv2.imwrite(os.path.join(out_dir, out_name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

def crop_standardize_imgs_files(bboxs_txt, imgs_dir, out_dir, out_size=(224, 224), n_procs=16):
    df = pd.read_csv(bboxs_txt, engine="python", sep='\ +', skiprows=1, index_col=0, dtype={0:str, 1:np.int32, 2:np.int32, 3:np.int32, 4:np.int32})
    os.makedirs(out_dir, exist_ok=True)

    df.index = [os.path.join(imgs_dir, index) for index in df.index]
    with Pool(n_procs) as pool:
        _ = pool.starmap(__crop_standardize_img__, [(index, row.values, out_dir, out_size) for index, row in df.iterrows()])



# Normalizes image from 0 to 255 and make the values integer
def normalize(image):
    image_t = ((image - np.min(image)) * 255 / (np.max(image) - np.min(image))).astype(np.uint8)
    return image_t

"""
Returns all filenames from a given directory
"""
def getFilenamesFromDir(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))
    return files



def get_best_exps(top_dir, top_k=3, metric="Mean Bbox Iou", metric_set="Val"):
    dfs = glob(os.path.join(top_dir, "*", "metrics.csv"))
    dfs = [(df, pd.read_csv(df)) for df in dfs]
    dfs = [(filename, df[df["Set"]==metric_set][metric].iloc[0]) for filename, df in dfs]
    dfs = sorted(dfs, key=lambda k : k[1], reverse=True)[:top_k]
    dfs = [(filename, pd.read_csv(filename)) for filename, _ in dfs]

    return dfs



if __name__ == '__main__':
    gen_bboxs_csv("../data/list_bbox_celeba_fixed.txt", "../data/Img/", bboxs_csv_outfile="bboxs.csv", return_df=False, n_procs=16)
    gen_bboxs_csv_x2y2("../data/list_bbox_celeba_fixed.txt", "../data/Img/", bboxs_csv_outfile="bboxs_x2y2.csv", return_df=False, n_procs=16)
    gen_splits_csv("~/Downloads/list_eval_partition.txt", splits_csv_outfile="splits.csv", return_df=False)
    gen_indvs_csv("~/Downloads/identity_CelebA.txt", indvs_csv_outfile="indvs.csv", return_df=False)
    crop_standardize_imgs_files("../data/list_bbox_celeba_fixed.txt", "../data/Img/", "Img_Crop_Resize", out_size=(224, 224), n_procs=16)