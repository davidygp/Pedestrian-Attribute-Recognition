import os
import pickle
import random

import numpy as np
import pandas as pd
from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)


def generate_data_description(pa100k_dir, peta_dir):
    """
    create a PA100K peta_dataset description file, which consists of images, labels
    """
    pa100k_data = loadmat(os.path.join(pa100k_dir, 'annotation.mat'))

    pa100k_dataset = EasyDict()
    pa100k_dataset.description = 'pa100k'
    # pa100k_dataset.root = os.path.join(pa100k_dir, 'data')

    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    pa100k_dataset.image_name = train_image_name + val_image_name + test_image_name

    pa100k_dataset.label = np.concatenate(
        (pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    assert pa100k_dataset.label.shape == (100000, 26)
    pa100k_dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]

    pa100k_df = pd.DataFrame(data=pa100k_dataset.label,
                             index=[i for i in range(pa100k_dataset.label.shape[0])],
                             columns=pa100k_dataset.attr_name)

    pa100k_df['image_id'] = pa100k_dataset.image_name
    print(pa100k_df.head())

    # annotated_pa100k_df = pd.read_csv(os.path.join(pa100k_dir, 'PA100K_program_annotated_zero_one_manual_4K.csv'))
    annotated_pa100k_df = pd.read_csv(os.path.join(pa100k_dir, 'program_annotated_zero_one_pa100k.csv'))

    annotated_pa100k_df = annotated_pa100k_df.drop(annotated_pa100k_df.columns[0], axis=1)

    annotated_pa100k_df['accessoryHat'] = pa100k_df['Hat']
    annotated_pa100k_df['accessorySunglasses'] = pa100k_df['Glasses']
    annotated_pa100k_df['carryingBackpack'] = pa100k_df['Backpack']
    annotated_pa100k_df['carryingFolder'] = pa100k_df['HandBag']
    annotated_pa100k_df['carryingMessengerBag'] = pa100k_df['ShoulderBag']
    annotated_pa100k_df['carryingOther'] = pa100k_df['HoldObjectsInFront']
    annotated_pa100k_df['footwearBoots'] = pa100k_df['boots']
    annotated_pa100k_df['lowerBodyShortSkirt'] = pa100k_df['Skirt&Dress']
    annotated_pa100k_df['lowerBodyShorts'] = pa100k_df['Shorts']
    annotated_pa100k_df['lowerBodyTrousers'] = pa100k_df['Trousers']
    annotated_pa100k_df['personalLarger60'] = pa100k_df['AgeOver60']
    annotated_pa100k_df['personalLess15'] = pa100k_df['AgeLess18']
    annotated_pa100k_df['personalLess60'] = pa100k_df['Age18-60']
    annotated_pa100k_df['upperBodyCasual'] = pa100k_df['UpperPlaid']
    annotated_pa100k_df['upperBodyFormal'] = pa100k_df['LongCoat']
    annotated_pa100k_df['upperBodyLogo'] = pa100k_df['UpperLogo']
    annotated_pa100k_df['upperBodyLongSleeve'] = pa100k_df['LongSleeve']
    annotated_pa100k_df['upperBodyOther'] = pa100k_df['UpperStride'] | pa100k_df['UpperSplice']
    annotated_pa100k_df['upperBodyShortSleeve'] = pa100k_df['ShortSleeve']

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

    combined_pa100k_peta_df = pd.concat([annotated_pa100k_df, new_peta_df])

    # combined_pa100k_peta_df.to_csv('./combined_pa100k_peta_117_columns_original.csv')
    # combined_pa100k_peta_df.to_csv('./combined_pa100k_peta_117_columns_validated.csv')

    # generate pkl file

    peta_pkl_file = os.path.join(peta_dir, 'dataset.pkl')

    # pa100k_peta_pkl_file = os.path.join(peta_dir, 'pa100k_4k_validated_peta_dataset.pkl')
    pa100k_peta_pkl_file = os.path.join(peta_dir, 'pa100k_origin_peta_dataset.pkl')

    with open(peta_pkl_file, 'rb') as handle:
        peta_data_info = pickle.load(handle)

        peta_data_info.image_name = combined_pa100k_peta_df['image_id'].tolist()
        combined_pa100k_peta_df.drop(['image_id'], inplace=True, axis=1)
        peta_data_info.label = combined_pa100k_peta_df.values
        peta_data_info.attr_name = combined_pa100k_peta_df.columns
        peta_data_info.partition.train = []
        peta_data_info.partition.val = []
        peta_data_info.partition.trainval = []
        peta_data_info.partition.test = []

        list_100k = [x for x in range(100000)]
        list_19000 = [x for x in range(19000)]
        random.shuffle(list_100k)
        random.shuffle(list_19000)

        for idx in range(5):
            test = np.array(list_19000[-7600:])

            train = np.array(list_100k[:80000] + list_19000[:11400])
            val = np.array(list_100k[80000:])

            trainval = np.array(list_100k[:80000] + list_19000[:11400] + list_100k[80000:])

            peta_data_info.partition.train.append(train)
            peta_data_info.partition.val.append(val)
            peta_data_info.partition.trainval.append(trainval)
            peta_data_info.partition.test.append(test)

            weight_train = np.mean(peta_data_info.label[train], axis=0)
            weight_trainval = np.mean(peta_data_info.label[trainval], axis=0)

            peta_data_info.weight_train.append(weight_train)
            peta_data_info.weight_trainval.append(weight_trainval)

    with open(pa100k_peta_pkl_file, 'wb') as handle:
        pickle.dump(peta_data_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pass


def rename_peta_images(peta_dir):
    image_dir = os.path.join(peta_dir, 'images')
    for filename in os.listdir(image_dir):
        new_name = f'{100000 + int(filename[:-4]) :06}.png'

        os.rename(os.path.join(image_dir, filename), os.path.join(image_dir, new_name))

        pass


if __name__ == "__main__":
    pa100k_dir = './data/PA100k/'
    peta_dir = './data/PETA/'
    generate_data_description(pa100k_dir, peta_dir)

    # rename_peta_images(peta_dir)
