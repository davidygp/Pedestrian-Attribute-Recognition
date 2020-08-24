from pedestrian_attri_recog_model import AttrRecogModel
import os
import pandas as pd

image_data_path = 'D:\\PA-100K\\PA-100K-20200420T144034Z-001\\PA-100K\\data\\release_data\\release_data\\'


def annotate_images_under_directory(image_data_path):
    model = AttrRecogModel()

    image_files = [f for f in os.listdir(image_data_path) if f[-4:] == '.jpg']

    df = pd.DataFrame()

    for f in image_files:
        image_file = image_data_path + f
        result = model.predict_image_general(image_file)
        df2 = pd.DataFrame({k: [v] for k, v in result.items()})
        df2['image_id'] = f
        if df.empty:
            df = df2
        else:
            df = df.append(df2, ignore_index=True)

    df.to_csv(image_data_path + '_annotated.csv')


def convert_dataframe_to_zero_and_one():
    def change(x):
        if x < 0.5:
            return 0
        else:
            return 1

    data = pd.read_csv(image_data_path + '_annotated.csv')
    data.drop(data.columns[[0]], axis=1, inplace=True)

    headers = data.columns.to_list()
    headers.pop()

    for i in range(len(headers)):
        data[headers[i]] = data[headers[i]].apply(change)

    data.to_csv(image_data_path + '_annotated_zero_one.csv')


convert_dataframe_to_zero_and_one()
