#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 23:20:41 2020

@author: jiahao
"""


import pandas as pd
import os
import shutil

df = pd.read_csv('./working_code/RAPv2.csv')

peta_attr = ["personalLess30", "personalLess45", "personalLess60", "personalLarger60", "carryingBackpack", "carryingOther", "lowerBodyCasual", "upperBodyCasual", "lowerBodyFormal", "upperBodyFormal", "accessoryHat", "upperBodyJacket", "lowerBodyJeans", "footwearLeatherShoes", "upperBodyLogo", "hairLong", "personalMale", "carryingMessengerBag", "accessoryMuffler", "accessoryNothing", "carryingNothing", "upperBodyPlaid", "carryingPlasticBags", "footwearSandals", "footwearShoes", "lowerBodyShorts", "upperBodyShortSleeve", "lowerBodyShortSkirt", "footwearSneaker", "upperBodyThinStripes", "accessorySunglasses", "lowerBodyTrousers", "upperBodyTshirt", "upperBodyOther", "upperBodyVNeck", "upperBodyBlack", "upperBodyBlue", "upperBodyBrown", "upperBodyGreen", "upperBodyGrey", "upperBodyOrange", "upperBodyPink", "upperBodyPurple", "upperBodyRed", "upperBodyWhite", "upperBodyYellow", "lowerBodyBlack", "lowerBodyBlue", "lowerBodyBrown", "lowerBodyGreen", "lowerBodyGrey", "lowerBodyOrange", "lowerBodyPink", "lowerBodyPurple", "lowerBodyRed", "lowerBodyWhite", "lowerBodyYellow", "hairBlack", "hairBlue", "hairBrown", "hairGreen", "hairGrey", "hairOrange", "hairPink", "hairPurple", "hairRed", "hairWhite", "hairYellow", "footwearBlack", "footwearBlue", "footwearBrown", "footwearGreen", "footwearGrey", "footwearOrange", "footwearPink", "footwearPurple", "footwearRed", "footwearWhite", "footwearYellow", "accessoryHeadphone", "personalLess15", "carryingBabyBuggy", "hairBald", "footwearBoots", "lowerBodyCapri", "carryingShoppingTro", "carryingUmbrella", "personalFemale", "carryingFolder", "accessoryHairBand", "lowerBodyHotPants", "accessoryKerchief", "lowerBodyLongSkirt", "upperBodyLongSleeve", "lowerBodyPlaid", "lowerBodyThinStripes", "carryingLuggageCase", "upperBodyNoSleeve", "hairShort", "footwearStocking", "upperBodySuit", "carryingSuitcase", "lowerBodySuits", "upperBodySweater", "upperBodyThickStripes", "carryingBlack", "carryingBlue", "carryingBrown", "carryingGreen", "carryingGrey", "carryingOrange", "carryingPink", "carryingPurple", "carryingRed", "carryingWhite", "carryingYellow"]

#%% 
attr_map =  {#JH's attributes
             'hs-Mask':'accessoryFaceMask',
             'hs-Muffler':'accessoryKerchief',
             'shoes-ColorBlue':'footwearBlue',
             'shoes-ColorOrange':'footwearOrange',
             'shoes-ColorPurple':'footwearPurple',
             'lb-ColorGreen':'lowerBodyGreen',
             'lb-ColorOrange':'lowerBodyOrange',
             'lb-ColorRed':'lowerBodyRed',
             'lb-ColorYellow':'lowerBodyYellow',
             'AgeLess16':'personalLess15',
    
             #DY's attributes
             'shoes-ColorGreen':'footwearGreen',
             'shoes-ColorYellow':'footwearYellow',
             'lb-ColorPurple':'lowerBodyPurple',
             'lb-ColorPink':'lowerBodyPink',
             'shoes-ColorPink':'footwearPink'
             }

# check available data
for attr in attr_map.keys():
    x = df[df[attr]==1].count()[attr]
    print(attr, ": ", x)
    
#%% save rare attribute data

# Get only the desired attributes
df = df[["Image"] + list(attr_map.keys())]

# put all attribute label to 2
for attr in peta_attr:
    df[attr] = 2

# change the rare attribute to RAP label
for key, val in attr_map.items():
    df[val] = df[key]

# df["image_id"] = [str(x) + ".png" for x in list(np.arange(len(df)) + 119001)]

output_df = df[peta_attr + ["Image"]]

#%% output
output_df.to_csv("RAP_rare_labelled.csv", header=True, index=False)
