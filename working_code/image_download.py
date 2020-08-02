#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:37:03 2020

@author: jiahao
"""

#%% define functions
from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()

def downloadimages(query):
    arguments = {'keywords': query, #keywords is the search query
                'format': 'jpg', #image format
                'limit': 99, #the number of images to be downloaded
                'print_urls': True}
    
    try:
        response.download(arguments)
        
    #handling file not found error
    except FileNotFoundError:
        arguments = {"keywords": query, 
                     "format": "jpg", 
                     "limit":99, 
                     "print_urls":True} 
                       
        # Providing arguments for the searched query 
        try: 
            # Downloading the photos based 
            # on the given arguments 
            response.download(arguments)  
        except: 
            pass
        
#%% download imaged
search_queries = ['pedestrian red hair']

for query in search_queries:
    downloadimages(query)
    print()

#%% Standardize Image Format and Size
from PIL import Image
import os.path
import glob

def convertjpg(jpgfile,outdir,width=512,height=512):
    img = Image.open(jpgfile)
    img = img.convert('RGB')
    try:
        new_img = img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


for jpgfile in glob.glob("C:/Users/Yazhi/Desktop/Test/*.jpg"):
    convertjpg(jpgfile,"C:/Users/Yazhi/Desktop/Test")
    print('Convert Successfully.')
    
#%% Standardize Image Name
import os
os.getcwd()
collection = "D:/NUS/semester 2/CA2 Food"
for i, filename in enumerate(os.listdir(collection)):
    os.rename("D:/NUS/semester 2/CA2 Food/" + filename, 
              "D:/NUS/semester 2/CA2 Food/" + str(i) + " - food.jpg")
