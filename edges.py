#!/usr/bin/env python3
# Author: Piero Toffanin
# License: AGPLv3

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


import rasterio
import numpy as np
import numpy.ma as ma
import multiprocessing
import argparse
from scipy import ndimage
import functools
from opensfm import dataset
from rasterio.windows import Window
import cv2
from math import sqrt
from skimage.draw import line

#default_dem_path = "odm_dem/dsm.tif"
default_dem_path = "odm_dem/mesh_dsm.tif"
default_outdir = "sobel_tests"
default_image_list = "img_list.txt"

#dataset_path = "/datasets/cmparks"
dataset_path = "/datasets/brighton2"
dem_path = os.path.join(dataset_path, default_dem_path)
interpolation = "nearest"
with_alpha = True
image_list = os.path.join(dataset_path, default_image_list)

cwd_path = os.path.join(dataset_path, default_outdir)
if not os.path.exists(cwd_path):
    os.makedirs(cwd_path)


# with open(image_list) as f:
#     target_images = [img + ".tif" for img in list(filter(lambda filename: filename != '', map(str.strip, f.read().split("\n"))))]
target_images = ["DJI_0030.JPG.tif"]

print("Processing %s images" % len(target_images))

if not os.path.exists(dem_path):
    print("Whoops! %s does not exist. Provide a path to a valid DEM" % dem_path)
    exit(1)


# Read DEM
print("Reading DEM: %s" % dem_path)
with rasterio.open(dem_path) as dem_raster:
    dem = dem_raster.read()[0]
    dem_has_nodata = dem_raster.profile.get('nodata') is not None

    if dem_has_nodata:
        m = ma.array(dem, mask=dem==dem_raster.nodata)
        dem_min_value = m.min()
        dem_max_value = m.max()
    else:
        dem_min_value = dem.min()
        dem_max_value = dem.max()

    print("DEM Minimum: %s" % dem_min_value)
    print("DEM Maximum: %s" % dem_max_value)
    
    h, w = dem.shape

    crs = dem_raster.profile.get('crs')
    transform = dem_raster.profile.get('transform')
    dem_offset_x, dem_offset_y = (0, 0)

    if crs:
        print("DEM has a CRS: %s" % str(crs))

        # Read coords.txt
        coords_file = os.path.join(dataset_path, "odm_georeferencing", "coords.txt")
        if not os.path.exists(coords_file):
            print("Whoops! Cannot find %s (we need that!)" % coords_file)
            exit(1)
        
        with open(coords_file) as f:
            line = f.readline() # discard

            # second line is a northing/easting offset
            line = f.readline().rstrip()
            dem_offset_x, dem_offset_y = map(float, line.split(" "))
        
        print("DEM offset: (%s, %s)" % (dem_offset_x, dem_offset_y))

    print("DEM dimensions: %sx%s pixels" % (w, h))
   
    # Read reconstruction
    udata = dataset.UndistortedDataSet(dataset.DataSet(os.path.join(dataset_path, "opensfm")), undistorted_data_path=os.path.join(dataset_path, "opensfm", "undistorted"))
    reconstructions = udata.load_undistorted_reconstruction()
    if len(reconstructions) == 0:
        raise Exception("No reconstructions available")

    max_workers = 4
    print("Using %s threads" % max_workers)

    reconstruction = reconstructions[0]

    canny_dem = np.full((h, w), 0, dtype=np.uint8)

    for shot in reconstruction.shots.values():
        if len(target_images) == 0 or shot.id in target_images:

            print("Processing %s..." % shot.id)

            shot_image = udata.load_undistorted_image(shot.id)
            gray_image = shot_image[:,:,2]
            #gray_image = cv2.cvtColor(shot_image, cv2.COLOR_RGB2GRAY)
            #gray_image = cv2.bilateralFilter(gray_image,15,75,75)

            median = np.median(gray_image)
            sigma = 0.33
            lower_thresh = int(max(0, (1.0 - sigma) * median))
            upper_thresh = int(min(255, (1.0 + sigma) * median))

            canny = cv2.Canny(gray_image, lower_thresh, upper_thresh)
            canny = ndimage.maximum_filter(canny, size=7, mode='nearest')

            import lsd
            lines = lsd.line_segment_detector(gray_image)
            line_image = np.copy(gray_image) * 0 
            for line in lines:
                x1, y1, x2, y2, cx, cy, l, w = line
                cv2.line(line_image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0), 1)

            cv2.imwrite(os.path.join(cwd_path, "lines.jpg"), line_image)
            exit(1)

            # rho = 1  # distance resolution in pixels of the Hough grid
            # theta = np.pi / 180  # angular resolution in radians of the Hough grid
            # threshold = 15  # minimum number of votes (intersections in Hough grid cell)
            # min_line_length = 20  # minimum number of pixels making up a line
            # max_line_gap = 20  # maximum gap in pixels between connectable line segments
            # line_image = np.copy(canny) * 0  # creating a blank to draw lines on

            # # Run Hough on edge detected image
            # # Output "lines" is an array containing endpoints of detected line segments
            # lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]),
            #                     min_line_length, max_line_gap)

            # for line in lines:
            #     for x1,y1,x2,y2 in line:
            #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)
            
            r = shot.pose.get_rotation_matrix()
            Xs, Ys, Zs = shot.pose.get_origin()
            cam_grid_y, cam_grid_x = dem_raster.index(Xs + dem_offset_x, Ys + dem_offset_y)


            a1 = r[0][0]
            b1 = r[0][1]
            c1 = r[0][2]
            a2 = r[1][0]
            b2 = r[1][1]
            c2 = r[1][2]
            a3 = r[2][0]
            b3 = r[2][1]
            c3 = r[2][2]

            distance_map = np.full((h, w), np.nan)

            for j in range(0, h):
                for i in range(0, w):
                    distance_map[j][i] = sqrt((cam_grid_x - i) ** 2 + (cam_grid_y - j) ** 2)
            distance_map[distance_map==0] = 1e-7

            print("Camera pose: (%f, %f, %f)" % (Xs, Ys, Zs))

            img_h, img_w, num_bands = shot_image.shape
            print("Image dimensions: %sx%s pixels" % (img_w, img_h))
            f = shot.camera.focal * max(img_h, img_w)
            has_nodata = dem_raster.profile.get('nodata') is not None


            def process_pixels(step):
                imgout = np.full((dem_bbox_h, dem_bbox_w), np.nan, dtype=np.uint8)
                minx = dem_bbox_w
                miny = dem_bbox_h
                maxx = 0
                maxy = 0

                for j in range(dem_bbox_miny, dem_bbox_maxy + 1):
                    if j % max_workers == step:
                        im_j = j - dem_bbox_miny

                        for i in range(dem_bbox_minx, dem_bbox_maxx + 1):
                            im_i = i - dem_bbox_minx

                            # World coordinates
                            Xa, Ya = dem_raster.xy(j, i)
                            Za = dem[j][i]

                            # Skip nodata
                            if has_nodata and Za == dem_raster.nodata:
                                continue

                            # Remove offset (our cameras don't have the geographic offset)
                            Xa -= dem_offset_x
                            Ya -= dem_offset_y

                            # Colinearity function http://web.pdx.edu/~jduh/courses/geog493f14/Week03.pdf
                            dx = (Xa - Xs)
                            dy = (Ya - Ys)
                            dz = (Za - Zs)

                            den = a3 * dx + b3 * dy + c3 * dz
                            x = (img_w - 1) / 2.0 - (f * (a1 * dx + b1 * dy + c1 * dz) / den)
                            y = (img_h - 1) / 2.0 - (f * (a2 * dx + b2 * dy + c2 * dz) / den)

                            if x >= 0 and y >= 0 and x <= img_w - 1 and y <= img_h - 1:
                                check_dem_points = np.column_stack(line(i, j, cam_grid_x, cam_grid_y))
                                check_dem_points = check_dem_points[np.all(np.logical_and(np.array([0, 0]) <= check_dem_points, check_dem_points < [w, h]), axis=1)]

                                visible = True
                                for p in check_dem_points:
                                    ray_z = Zs + (distance_map[p[1]][p[0]] / distance_map[j][i]) * dz
                                    if ray_z > dem_max_value:
                                        break

                                    if dem[p[1]][p[0]] > ray_z:
                                        visible = False
                                        break
                                if not visible:
                                    continue

                                # nearest
                                xi = img_w - 1 - int(round(x))
                                yi = img_h - 1 - int(round(y))
                                values = canny[yi][xi]

                                # We don't consider all zero values (pure black)
                                # to be valid sample values. This will sometimes miss
                                # valid sample values.

                                if not np.all(values == 0):
                                    minx = min(minx, im_i)
                                    miny = min(miny, im_j)
                                    maxx = max(maxx, im_i)
                                    maxy = max(maxy, im_j)

                                    imgout[im_j][im_i] = 0 if canny[yi][xi] == 0 else 1
                return (imgout, (minx, miny, maxx, maxy))

            # Compute bounding box of image coverage
            # assuming a flat plane at Z = min Z
            # (Otherwise we have to scan the entire DEM)
            # The Xa,Ya equations are just derived from the colinearity equations
            # solving for Xa and Ya instead of x,y
            def dem_coordinates(cpx, cpy):
                """
                :param cpx principal point X (image coordinates)
                :param cpy principal point Y (image coordinates)
                """
                Za = dem_min_value
                m = (a3*b1*cpy - a1*b3*cpy - (a3*b2 - a2*b3)*cpx - (a2*b1 - a1*b2)*f)
                Xa = dem_offset_x + (m*Xs + (b3*c1*cpy - b1*c3*cpy - (b3*c2 - b2*c3)*cpx - (b2*c1 - b1*c2)*f)*Za - (b3*c1*cpy - b1*c3*cpy - (b3*c2 - b2*c3)*cpx - (b2*c1 - b1*c2)*f)*Zs)/m
                Ya = dem_offset_y + (m*Ys - (a3*c1*cpy - a1*c3*cpy - (a3*c2 - a2*c3)*cpx - (a2*c1 - a1*c2)*f)*Za + (a3*c1*cpy - a1*c3*cpy - (a3*c2 - a2*c3)*cpx - (a2*c1 - a1*c2)*f)*Zs)/m

                y, x = dem_raster.index(Xa, Ya)
                return (x, y)

            dem_ul = dem_coordinates(-(img_w - 1) / 2.0, -(img_h - 1) / 2.0)
            dem_ur = dem_coordinates((img_w - 1) / 2.0, -(img_h - 1) / 2.0)
            dem_lr = dem_coordinates((img_w - 1) / 2.0, (img_h - 1) / 2.0)
            dem_ll = dem_coordinates(-(img_w - 1) / 2.0, (img_h - 1) / 2.0)
            dem_bbox = [dem_ul, dem_ur, dem_lr, dem_ll]
            dem_bbox_x = np.array(list(map(lambda xy: xy[0], dem_bbox)))
            dem_bbox_y = np.array(list(map(lambda xy: xy[1], dem_bbox)))

            dem_bbox_minx = min(w - 1, max(0, dem_bbox_x.min()))
            dem_bbox_miny = min(h - 1, max(0, dem_bbox_y.min()))
            dem_bbox_maxx = min(w - 1, max(0, dem_bbox_x.max()))
            dem_bbox_maxy = min(h - 1, max(0, dem_bbox_y.max()))
            
            dem_bbox_w = 1 + dem_bbox_maxx - dem_bbox_minx
            dem_bbox_h = 1 + dem_bbox_maxy - dem_bbox_miny

            print("Iterating over DEM box: [(%s, %s), (%s, %s)] (%sx%s pixels)" % (dem_bbox_minx, dem_bbox_miny, dem_bbox_maxx, dem_bbox_maxy, dem_bbox_w, dem_bbox_h))

            if max_workers > 1:
                with multiprocessing.Pool(max_workers) as p:
                    results = p.map(process_pixels, range(max_workers))
            else:
                results = [process_pixels(0)]

            results = list(filter(lambda r: r[1][0] <= r[1][2] and r[1][1] <= r[1][3], results))

            # Merge image
            imgout, _ = results[0]
            for j in range(dem_bbox_miny, dem_bbox_maxy + 1):
                im_j = j - dem_bbox_miny
                resimg, _ = results[j % max_workers]
                imgout[im_j] = resimg[im_j]

            # Merge bounds
            minx = dem_bbox_w
            miny = dem_bbox_h
            maxx = 0
            maxy = 0

            for _, bounds in results:
                minx = min(bounds[0], minx)
                miny = min(bounds[1], miny)
                maxx = max(bounds[2], maxx)
                maxy = max(bounds[3], maxy)

            if minx <= maxx and miny <= maxy:
                imgout = imgout[miny:maxy,minx:maxx]
                imgout[np.isnan(imgout)] = 0

                canny_dem[dem_bbox_miny + miny:dem_bbox_miny + miny + imgout.shape[0],
                          dem_bbox_minx + minx:dem_bbox_minx + minx + imgout.shape[1]] += imgout
    
    dem_transform = dem_raster.profile['transform']

    profile = {
        'driver': 'GTiff',
        'width': w,
        'height': h,
        'count': 1,
        'dtype': rasterio.dtypes.uint8,
        'transform': transform,
        'nodata': None,
        'crs': crs
    }

    outfile = os.path.join(cwd_path, "edges.tif")
    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(canny_dem, 1)

    print("Wrote %s" % outfile)

