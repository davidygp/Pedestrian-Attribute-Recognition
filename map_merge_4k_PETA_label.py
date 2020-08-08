import os
import pickle
import random

import numpy as np
import pandas as pd
from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)


def copy_4k_to_image_4k_peta_folder():
    from shutil import copyfile
    pa100k_dir = './data/PA100k/'
    peta_dir = './data/PETA/'
    image_dir = os.path.join(peta_dir, 'images')
    image_dir_4K_peata = os.path.join(peta_dir, 'images_4k_peta')

    annotated_4k_df = pd.read_csv(os.path.join(pa100k_dir, '4k_annotated_zero_one.csv'))
    annotated_4k_df = annotated_4k_df.drop(annotated_4k_df.columns[0], axis=1)
    image_4k_names = annotated_4k_df['image_id'].to_list()

    for image_name in image_4k_names:
        if len(image_name) == 9:
            image_name = '0' + image_name
        print(image_name)
        copyfile(os.path.join(image_dir, image_name), os.path.join(image_dir_4K_peata, image_name))

    for i in range(19000):
        name = f'{100001 + i :06}.png'
        copyfile(os.path.join(image_dir, name), os.path.join(image_dir_4K_peata, name))


def generate_data_description(pa100k_dir, peta_dir):

    annotated_4k_df = pd.read_csv(os.path.join(pa100k_dir, '4k_annotated_zero_one.csv'))
    annotated_4k_df = annotated_4k_df.drop(annotated_4k_df.columns[0], axis=1)
    annotated_4k_df['image_id'] = annotated_4k_df['image_id'].apply(lambda x: '0' + x if len(x) == 9 else x)

    # from sklearn.utils import shuffle
    # annotated_4k_df = shuffle(annotated_4k_df)
    image_4k_names = annotated_4k_df['image_id'].to_list()

    """
    create a PETA peta_dataset description file, which consists of images, labels
    """
    peta_data = loadmat(os.path.join(peta_dir, 'PETA_added.mat'))
    peta_dataset = EasyDict()
    peta_dataset.description = 'peta'
    peta_dataset.reorder = 'group_order'
    peta_dataset.root = os.path.join(peta_dir, 'images')
    peta_dataset.image_name = [f'{100000 + i + 1:06}.png' for i in range(19000)]

    raw_attr_name = [i[0][0].strip() for i in peta_data['peta'][0][0][1]]
    raw_label = peta_data['peta'][0][0][0][:, 4:]

    peta_dataset.label = raw_label
    peta_dataset.attr_name = raw_attr_name

    new_peta_image = pd.Series(peta_dataset.image_name)
    new_peta_df = pd.DataFrame(data=raw_label[0:, 0:], columns=peta_dataset.attr_name)
    new_peta_df['image_id'] = new_peta_image

    combined_4k_peta_df = pd.concat([annotated_4k_df, new_peta_df])

    # combined_4k_peta_df.to_csv('./combined_pa100k_peta_117_columns_original.csv')

    # generate pkl file

    peta_pkl_file = os.path.join(peta_dir, 'dataset.pkl')

    pa4k_peta_pkl_file = os.path.join(peta_dir, 'pa_4k_peta_dataset.pkl')
    with open(peta_pkl_file, 'rb') as handle:
        peta_data_info = pickle.load(handle)

        peta_data_info.image_name = combined_4k_peta_df['image_id'].tolist()
        combined_4k_peta_df.drop(['image_id'], inplace=True, axis=1)
        peta_data_info.label = combined_4k_peta_df.values
        peta_data_info.attr_name = combined_4k_peta_df.columns
        peta_data_info.partition.train = []
        peta_data_info.partition.val = []
        peta_data_info.partition.trainval = []
        peta_data_info.partition.test = []

        list_4k = [x for x in range(4000)]
        list_19000 = [x for x in range(19000)]
        random.shuffle(list_4k)
        random.shuffle(list_19000)

        for idx in range(5):
            test = np.array(list_19000[-7600:])

            train = np.array(list_4k + list_19000[:11400])
            val = np.array([])

            trainval = np.array(list_4k + list_19000[:11400])

            peta_data_info.partition.train.append(train)
            peta_data_info.partition.val.append(val)
            peta_data_info.partition.trainval.append(trainval)
            peta_data_info.partition.test.append(test)

            weight_train = np.mean(peta_data_info.label[train], axis=0)
            weight_trainval = np.mean(peta_data_info.label[trainval], axis=0)

            peta_data_info.weight_train.append(weight_train)
            peta_data_info.weight_trainval.append(weight_trainval)

    with open(pa4k_peta_pkl_file, 'wb') as handle:
        pickle.dump(peta_data_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pass


if __name__ == "__main__":
    pa100k_dir = './data/PA100k/'
    peta_dir = './data/PETA/'
    generate_data_description(pa100k_dir, peta_dir)

    # copy_4k_to_image_4k_peta_folder()
    # rename_peta_images(peta_dir)
