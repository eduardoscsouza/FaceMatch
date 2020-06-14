import numpy as np
import pandas as pd
import imageio
import os



def gen_bboxs_csv(bboxs_txt, imgs_dir, bboxs_csv_outfile="bboxs.csv", return_df=False):
    df = pd.read_csv(bboxs_txt, engine="python", sep='\ +', skiprows=1, index_col=0, dtype={0:str, 1:np.int32, 2:np.int32, 3:np.int32, 4:np.int32})

    imgs_shapes = [imageio.imread(os.path.join(imgs_dir, img)).shape for img in df.index]
    heights, widths = np.asarray([shape[0] for shape in imgs_shapes]), np.asarray([shape[1] for shape in imgs_shapes])
    df["y_1"], df["height"] = df["y_1"] / heights, df["height"] / heights
    df["x_1"], df["width"] = df["x_1"] / widths, df["width"] / widths

    df.to_csv(bboxs_csv_outfile)
    if return_df:
        return df

def get_bboxs_df(bboxs_csv):
    return pd.read_csv("bboxs.csv", index_col=0)


if __name__ == "__main__":
    gen_bboxs_csv("../CelebA/Anno/list_bbox_celeba.txt", "../CelebA/Img/", "../data/labels/bboxs.csv")
    '''
    os.makedirs("temp/or/", exist_ok=True)
    os.makedirs("temp/crop/", exist_ok=True)

    gen_bboxs_csv("../CelebA/Anno/list_bbox_celeba.txt", "../CelebA/Img/", return_df=True)
    df = get_bboxs_df("bboxs.csv")
    for df_index, df_row in df.iterrows():
        l_img = imageio.imread(os.path.join("../CelebA/Img/", df_index))

        df_row = df_row.values
        df_row[[0, 2]] *= l_img.shape[1]
        df_row[[1, 3]] *= l_img.shape[0]

        sli_i = slice(int(df_row[1]), int(df_row[1]+df_row[3]))
        sli_j = slice(int(df_row[0]), int(df_row[0]+df_row[2]))
        cut = l_img[sli_i, sli_j]
        print(cut.shape)

        imageio.imwrite("temp/or/{}".format(df_index), l_img)
        imageio.imwrite("temp/crop/{}".format(df_index), cut)
    '''