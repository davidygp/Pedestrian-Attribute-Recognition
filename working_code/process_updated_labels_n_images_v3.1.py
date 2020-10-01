#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import glob
import pandas as pd
import math
import numpy as np
import os
import scipy.io as sio

from PIL import Image

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)


# In[2]:


# The order matters, don't touch these #
previous_attributes = ['personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'carryingBackpack',                        'carryingOther', 'lowerBodyCasual', 'upperBodyCasual', 'lowerBodyFormal', 'upperBodyFormal',                        'accessoryHat', 'upperBodyJacket', 'lowerBodyJeans', 'footwearLeatherShoes', 'upperBodyLogo',                        'hairLong', 'personalMale', 'carryingMessengerBag', 'accessoryMuffler', 'accessoryNothing',                        'carryingNothing', 'upperBodyPlaid', 'carryingPlasticBags', 'footwearSandals', 'footwearShoes',                        'lowerBodyShorts', 'upperBodyShortSleeve', 'lowerBodyShortSkirt', 'footwearSneaker',                        'upperBodyThinStripes', 'accessorySunglasses', 'lowerBodyTrousers', 'upperBodyTshirt',                        'upperBodyOther', 'upperBodyVNeck', 'upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown',                        'upperBodyGreen', 'upperBodyGrey', 'upperBodyOrange', 'upperBodyPink', 'upperBodyPurple',                        'upperBodyRed', 'upperBodyWhite', 'upperBodyYellow', 'lowerBodyBlack', 'lowerBodyBlue',                        'lowerBodyBrown', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyOrange', 'lowerBodyPink',                        'lowerBodyPurple', 'lowerBodyRed', 'lowerBodyWhite', 'lowerBodyYellow', 'hairBlack', 'hairBlue',                        'hairBrown', 'hairGreen', 'hairGrey', 'hairOrange', 'hairPink', 'hairPurple', 'hairRed',                        'hairWhite', 'hairYellow', 'footwearBlack', 'footwearBlue', 'footwearBrown', 'footwearGreen',                        'footwearGrey', 'footwearOrange', 'footwearPink', 'footwearPurple', 'footwearRed', 'footwearWhite',                        'footwearYellow', 'accessoryHeadphone', 'personalLess15', 'carryingBabyBuggy', 'hairBald',                        'footwearBoots', 'lowerBodyCapri', 'carryingShoppingTro', 'carryingUmbrella', 'personalFemale',                        'carryingFolder', 'accessoryHairBand', 'lowerBodyHotPants', 'accessoryKerchief',                        'lowerBodyLongSkirt', 'upperBodyLongSleeve', 'lowerBodyPlaid', 'lowerBodyThinStripes',                        'carryingLuggageCase', 'upperBodyNoSleeve', 'hairShort', 'footwearStocking', 'upperBodySuit',                        'carryingSuitcase', 'lowerBodySuits', 'upperBodySweater', 'upperBodyThickStripes']


added_attributes = ['carryingBlack', 'carryingBlue', 'carryingBrown', 'carryingGreen', 'carryingGrey',                 'carryingOrange', 'carryingPink', 'carryingPurple', 'carryingRed', 'carryingWhite', 'carryingYellow']

PETA_folders = ["3DPeS", "CAVIAR4REID", "CUHK", "GRID", "i-LID", "MIT", "PRID", "SARC3D", "TownCentre", "VIPeR"]
# The order matters, don't touch these #


# Change here #
labels_text_filepaths = "./Updated_Labels/"
peta_dataset_filepaths = "./PETA dataset/"

peta_images_output_filepath="./data/PETA/images/"
process_images = True
output_mat_filepath = './data/PETA/PETA.mat'

prev_peta_filepath = "./PETA_old.mat"
# Change here #


# In[3]:


assert(os.path.exists(labels_text_filepaths)), "We need the updated labels in the 3DPeS.txt, CAVIAR4REID.txt, ... format"
assert(os.path.exists(peta_dataset_filepaths)), "Please download the original PETA data with the following file format: PETA dataset\{3DPeS, CAVAIR4REID, ...}\archive, it can be found at http://mmlab.ie.cuhk.edu.hk/projects/PETA.html#:~:text=The%20PETA%20dataset%20consists%20of,%23Images"

assert(os.path.exists(prev_peta_filepath)), "Please download the original PETA.mat file and rename it to %s." %(prev_peta_filepath)


# In[4]:


if process_images and not os.path.exists(peta_images_output_filepath):
    print("Creating peta image output filepath at %s" %(peta_images_output_filepath))
    os.makedirs(peta_images_output_filepath)


# # Putting all the attributes together

# In[5]:


all_attributes = previous_attributes + added_attributes


# In[6]:


all_attributes1 = np.zeros((len(all_attributes),1), dtype=np.object)


# In[7]:


all_attributes1[:] = [[x] for x in all_attributes]


# # Process the labels

# In[8]:


"""
# Note this doesn't work with mac/linux because of sometimes additional files, and its sorted differently
all_label_filepath = glob.glob(labels_text_filepaths + "*")
sorted(all_label_filepath)
all_label_filepath
"""
all_label_filepath = [os.path.join(labels_text_filepaths, PETA_folder) + ".txt" for PETA_folder in PETA_folders]
print(all_label_filepath)


# In[9]:


attribute_labels_df = pd.DataFrame()

dataset_id = 1
for label_filepath in all_label_filepath:
    #print(label_filepath)
    with open(label_filepath, 'r') as f:
        file = f.readlines()

    df_tmp = pd.DataFrame(file, columns = ['temp']) 
    df_tmp['dataset_id'] = dataset_id
    #print(df_tmp.shape)

    attribute_labels_df = pd.concat([attribute_labels_df, df_tmp])
    #print(df.shape)
    
    dataset_id += 1

del df_tmp


# In[10]:


print(attribute_labels_df.shape)


# In[11]:


print(attribute_labels_df.head())


# In[12]:


attribute_labels_df["pedestrian_id"] = attribute_labels_df.apply(lambda x: int(x["temp"].split(" ")[0].split(".")[0]), axis=1)


# In[13]:


# Extract the one-hot-encoding based on the order in the all_attributes (to fit the original mat file)
for attr in all_attributes:
    attribute_labels_df[attr] = attribute_labels_df.apply(lambda x: 1 if attr in x["temp"] else 0, axis=1)


# In[14]:


attribute_labels_df = attribute_labels_df.drop(columns="temp")


# # Process the images

# In[15]:


"""
# Note this doesn't work with mac/linux because of sometimes additional files, and its sorted differently
all_folders_filepath = glob.glob(peta_dataset_filepaths + "*")
sorted(all_folders_filepath)
all_folders_filepath
"""

all_folders_filepath = [os.path.join(peta_dataset_filepaths, PETA_folder) for PETA_folder in PETA_folders]
print(all_folders_filepath)


# In[16]:


# Contains the img_id, dataset_id, pedestrian_id
# img_id is a running number from 1 to n images
# dataset_id 1 to 10 based on the dataset, so 1 is 3DPeS, 10 is VIPeR, etc etc
# pedestrian_id is taken from the image name from the original PETA dataset
"""
['./PETA dataset\\3DPeS',
 './PETA dataset\\CAVIAR4REID',
 './PETA dataset\\CUHK',
 './PETA dataset\\GRID',
 './PETA dataset\\i-LID',
 './PETA dataset\\MIT',
 './PETA dataset\\PRID',
 './PETA dataset\\SARC3D',
 './PETA dataset\\TownCentre',
 './PETA dataset\\VIPeR']
"""
# for example if the naming is "100_3_FRAME_26_RGB.bmp" then the pedestrian_id is 100, this is to map to the attribute_labels_df

partial_ids_df = pd.DataFrame(columns = ['img_id', 'dataset_id', 'pedestrian_id'])


# In[17]:


img_id = 1

for i in range(len(all_folders_filepath)):
    images_filepath = glob.glob(all_folders_filepath[i] + "/archive/*[!txt]")
    for image_path in images_filepath:
        #ID = image_path.split("\\")[-1].split("_")[0].split(".")[0] 
        ID = os.path.split(image_path)[-1].split("_")[0].split(".")[0] # for compatibility btw windows/mac/linux
        #print(ID)
        
        if process_images:
            img = Image.open(image_path)
            img.save(peta_images_output_filepath + str(img_id).zfill(5)+".png")
        
        partial_ids_df.loc[img_id-1] = [int(img_id), int(i+1), int(ID)]
        img_id += 1


# In[18]:


print(partial_ids_df.shape)


# In[19]:


partial_ids_df = partial_ids_df.astype('int32')


# In[20]:


tmp = partial_ids_df[["dataset_id", "pedestrian_id"]].drop_duplicates().sort_values(by=["dataset_id", "pedestrian_id"]).reset_index()


# In[21]:


# So apparently there is also a unique ID for the dataset & pedestrian together.
tmp["dataset_pedestrian_id"] = tmp.index + 1
tmp = tmp.drop(columns="index")


# In[22]:


full_ids_df = pd.merge(partial_ids_df, tmp, on=["dataset_id", "pedestrian_id"], how="inner")


# In[24]:


print(full_ids_df.shape)


# In[25]:


del tmp


# # Combine the images and the labels

# In[26]:


attributes_n_ids_df = pd.merge(full_ids_df, attribute_labels_df, on=["dataset_id", "pedestrian_id"], how="inner")


# In[27]:


print(attributes_n_ids_df.shape)


# # Copy the train/test split from the original PETA.mat

# In[28]:


data = sio.loadmat(prev_peta_filepath)


# # Rearrange the columns

# In[29]:


ID_cols = ["img_id", "dataset_pedestrian_id", "dataset_id", "pedestrian_id"]

rearranged_columns = ID_cols + all_attributes

attributes_n_ids_df = attributes_n_ids_df[rearranged_columns]


# # Combine it all together 

# In[30]:


attribute_ids_n_values = np.array(attributes_n_ids_df.values) #, dtype='uint16')


# In[31]:


# Just look for last minute QC
#attribute_ids_n_values


# In[32]:


attribute_names = [[np.array([x], dtype="U")] for x in all_attributes]


# In[33]:


non_id_start_index = len(ID_cols) + 1


# In[34]:


attribute_train_ids = np.array([list(range(non_id_start_index, len(all_attributes) + non_id_start_index))])


# In[35]:


data_final = {'peta': {'data':attribute_ids_n_values,
                       'attribute':all_attributes1,
                       'selected_attribute':attribute_train_ids,
                       'partion_attribute':data['peta'][0][0][3]}} # this comes from the original PETA.mat


# In[36]:


sio.savemat(output_mat_filepath, data_final)


# In[37]:


# output_mat_filepath


# In[38]:


# loaddata = sio.loadmat(output_mat_filepath)


# In[39]:


# loaddata['peta'][0][0][0][:, 4:]


# In[40]:


# loaddata['peta'][0][0][0].shape


# In[41]:


# loaddata['peta'][0][0][0]


# In[42]:


# loaddata['peta'][0][0][0][18990]


# In[43]:


# import pickle


# In[44]:


# with open('./data/PETA/dataset.pkl', 'rb') as f:
#     x = pickle.load(f)


# In[45]:


# x.__dict__.keys()


# In[46]:


# x.label


# In[ ]:




