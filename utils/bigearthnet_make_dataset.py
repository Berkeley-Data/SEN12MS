#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json
import rasterio
from osgeo import gdal
import csv
import numpy as np
from glob import glob
import shutil

parser = argparse.ArgumentParser(
    description='This script reads the BigEarthNet image patches')
parser.add_argument('-r1', '--root_folder_s1', dest='root_folder_s1',
                    help='root folder path contains multiple patch folders of BigEarthNet-S1')
parser.add_argument('-r2', '--root_folder_s2', dest='root_folder_s2',
                    help='root folder path contains multiple patch folders of BigEarthNet-S2')
parser.add_argument('-splits', '---splits_folder', dest='splits_folder',
                    help='splits folder')
parser.add_argument('-output', '---output_folder', dest='output_folder',
                    help='Output folder')

args = parser.parse_args()

# Checks the existence of patch folders and populate the list of patch folder paths
folder_path_list = []
if args.root_folder_s1 and args.root_folder_s2:
    if not os.path.exists(args.root_folder_s1):
        print('ERROR: folder', args.root_folder_s1, 'does not exist')
        exit()
    elif not os.path.exists(args.root_folder_s2):
        print('ERROR: folder', args.root_folder_s2, 'does not exist')
        exit()
    #else:
        #folder_path_list = [i[0] for i in os.walk(args.root_folder_s1)][1:-1]
        #if len(folder_path_list) == 0:
        #    print('ERROR: there is no patch directories in the root folder')
        #    exit()
else:
    print('ERROR: -r arguments are required')
    exit()


# radar and spectral band names to read related GeoTIFF files in BIgEarthNet
band_names_s1 = ['VV', 'VH']
band_names_s2 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

def read_and_convert_bands(patch_folder_path, patch_name, band_names):
    tifs = []
    for band_name in band_names:
        # First finds related GeoTIFF path and reads values as an array
        band_path = os.path.join(
            patch_folder_path, patch_name + '_' + band_name + '.tif')
        #print("band_path",band_path)
        tifs.append(band_path)

    # /vsimem is special in-memory virtual "directory"
    outvrt = args.output_folder+'/stacked.vrt';
    if len(tifs) > 2:
        # Assume S2
        patch_output_folder = os.path.join(args.output_folder, 'S2', patch_name)
    else:
        raise TypeError("Only S2 image file names are allowed. S1 images are automatically retrieved")

    # Create the directory if it doesn't exist
    if not os.path.exists(patch_output_folder):
        os.makedirs(patch_output_folder)
    outtif = os.path.join(patch_output_folder, patch_name+'.tif')
    outds = gdal.BuildVRT(outvrt, tifs, separate=True)
    translateoptions = gdal.TranslateOptions(width=256, height=256, resampleAlg='cubic')
    outds = gdal.Translate(outtif, outds, options=translateoptions)

    # Copy json files to the corresponding folders
    src_json_file = os.path.join(patch_folder_path, patch_name+'_labels_metadata.json')
    dest_json_file = os.path.join(patch_output_folder, patch_name + '_labels_metadata.json')
    shutil.copy(src_json_file, dest_json_file)

    # Convert the mapping S1 image
    # Note: bigearthnet_train.csv,bigearthnet_test.csv,bigearthnet_val.csv has only S2 names
    s1_patch_name = s2_s1_mapping[patch_name]
    #print("S1 patch_name ", s1_patch_name)
    patch_folder_path = os.path.join(args.root_folder_s1, s1_patch_name)
    tifs = []
    for band_name in band_names_s1:
        # First finds related GeoTIFF path and reads values as an array
        band_path = os.path.join(
            patch_folder_path, s1_patch_name + '_' + band_name + '.tif')
        #print("band_path", band_path)
        tifs.append(band_path)
    if len(tifs) == 2:
        # Assume S1
        # NOte that we are using the S2 patch_name while writing the matching S1 patch.
        # That way, both S2 and S1 will have the same file name in their corresponding folders
        patch_output_folder = os.path.join(args.output_folder, 'S1', patch_name)
    else:
        raise TypeError("S1 image has more than 2 channels")
    # Create the directory if it doesn't exist
    if not os.path.exists(patch_output_folder):
        os.makedirs(patch_output_folder)
    outtif = os.path.join(patch_output_folder, patch_name + '.tif')
    outds = gdal.BuildVRT(outvrt, tifs, separate=True)
    translateoptions = gdal.TranslateOptions(width=256, height=256, resampleAlg='cubic')
    outds = gdal.Translate(outtif, outds, options=translateoptions)

    # Copy S1 json files to the corresponding folders
    src_json_file = os.path.join(patch_folder_path, s1_patch_name + '_labels_metadata.json')
    dest_json_file = os.path.join(patch_output_folder, patch_name + '_labels_metadata.json')
    shutil.copy(src_json_file, dest_json_file)

# Read the mapping file
s2_s1_mapping = {}
with open(args.splits_folder+"/bigearthnet_s1s2_mapping.csv", 'r') as fp:
    csv_reader = csv.reader(fp, delimiter=',')
    for row in csv_reader:
        s2_s1_mapping[row[0].strip()] = row[1].strip()
print("Cached S2->S1 mapping of length ", len(s2_s1_mapping))

# Read the train and test files
#csv_file_path_list = ['bigearthnet_train.csv', 'bigearthnet_test.csv']
csv_file_path_list = ['bigearthnet_test.csv']

patch_names_list = []
for csv_file in csv_file_path_list:
    splits = glob(f"{args.splits_folder}/{csv_file}")
    patch_names_list = []
    split_names = []
    for csv_file in splits:
        print("csv_file ",csv_file)
        split_names.append(os.path.basename(csv_file).split('.')[0])
        with open(csv_file, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                patch_names_list.append(row[0].strip())
                #print("Patch Name appended ",row[0].strip())

    # Read the spectral bands and convert them
    for patch_name in patch_names_list:
        #print("S2 patch_name ",patch_name)
        patch_folder_path = os.path.join(args.root_folder_s2, patch_name)
        read_and_convert_bands(patch_folder_path, patch_name, band_names_s2)


