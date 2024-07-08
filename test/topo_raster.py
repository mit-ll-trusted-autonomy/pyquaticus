import contextily as cx
import copy
import cv2
import mercantile as mt
import numpy as np
import shapely

from math import ceil
from PIL import Image
from scipy.ndimage import label
from timeit import default_timer as timer


COORD1 = (42.3066717, -71.0306793)
COORD2 = (42.3507779, -70.9488239)

def crop_tiles(img, ext, w, s, e, n, ll=True):
    """
    img : ndarray
        Image as a 3D array of RGB values
    ext : tuple
        Bounding box [minX, maxX, minY, maxY] of the returned image
    w : float
        West edge
    s : float
        South edge
    e : float
        East edge
    n : float
        North edge
    ll : Boolean
        [Optional. Default: True] If True, `w`, `s`, `e`, `n` are
        assumed to be lon/lat as opposed to Spherical Mercator.
    """
    #convert lat/lon bounds to Web Mercator XY (EPSG:3857)
    if ll:
        left, bottom = mt.xy(w, s)
        right, top = mt.xy(e, n)
    else:
        left, bottom = w, s
        right, top = e, n

    #determine crop
    X_size = ext[1] - ext[0]
    Y_size = ext[3] - ext[2]

    img_size_x = img.shape[1]
    img_size_y = img.shape[0]

    crop_start_x = ceil(img_size_x * (left - ext[0]) / X_size) - 1
    crop_end_x = ceil(img_size_x * (right - ext[0]) / X_size) - 1

    crop_start_y = ceil(img_size_y * (ext[2] - top) / Y_size)
    crop_end_y = ceil(img_size_y * (ext[2] - bottom) / Y_size) - 1

    #crop image
    image = img[
        crop_start_y : crop_end_y,
        crop_start_x : crop_end_x,
        :
    ]
    extent = (left, bottom, right, top)

    return image, extent


if __name__ == "__main__":
    # source = cx.providers.CartoDB.Voyager
    source = cx.providers.CartoDB.DarkMatterNoLabels

    print(source.get('attribution'))
    print()

    img, ext = cx.bounds2img(
        w=COORD1[1], s=COORD1[0], e=COORD2[1], n=COORD2[0], zoom='auto', source=source, ll=True,
        wait=0, max_retries=2, n_connections=1, use_cache=False, zoom_adjust=None
    )
    print("Initial image size:", img[:,:,:-1].shape)

    img, ext = crop_tiles(img[:,:,:-1], ext, COORD1[1], COORD1[0], COORD2[1], COORD2[0], ll=True)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("Final image size:", img.shape)
    print()

    image = Image.fromarray(img)
    image.save("topo_test.png")

    water = (42.3241868, -71.0223048)
    water_x, water_y  = mt.xy(water[1], water[0])

    left, bottom, right, top = ext

    water_pixel_x = ceil(img.shape[1] * (water_x - left)/(right - left)) - 1
    water_pixel_y = ceil(img.shape[0] * (top - water_y)/(top - bottom)) - 1

    water_pixel_color = img[water_pixel_y, water_pixel_x]
    mask = np.all(img == water_pixel_color, axis=-1)
    water_connectivity = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]]
    )
    labeled_mask, _ = label(mask, structure=water_connectivity)
    target_label = labeled_mask[water_pixel_y, water_pixel_x]
    land_mask = (labeled_mask == target_label) + (38 <= gray_img) * (gray_img <= 40)

    water_image = Image.fromarray(land_mask)
    water_image.save("land_mask.png")

    # water contours
    land_mask_binary = 255*land_mask.astype(np.uint8)
    water_contours, _ = cv2.findContours(land_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #https://docs.opencv.org/4.10.0/d4/d73/tutorial_py_contours_begin.html
    #https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html

    border_contour = max(water_contours, key=cv2.contourArea)
    border_land_mask = cv2.drawContours(np.zeros_like(land_mask_binary), [border_contour], -1, 255, -1)
    cv2.imwrite("border_land_mask.png", border_land_mask)

    # island contours
    water_mask = np.logical_not(land_mask)
    island_binary = 255*(border_land_mask * water_mask).astype(np.uint8)
    island_contours, _ = cv2.findContours(island_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    island_mask = cv2.drawContours(255*np.ones_like(island_binary), island_contours, -1, 0, -1)
    cv2.imwrite("island_mask.png", island_mask)

    #TODO: check outter contour to see if it is just the standard borders
    
    epsilon = 0.001

    #approximate outer contour (border land)
    eps = epsilon * cv2.arcLength(border_contour, True)
    border_cnt_approx = cv2.approxPolyDP(border_contour, eps, True)
    print(len(border_contour), "->", len(border_cnt_approx), "vertices")
    print()

    border_land_mask_approx = cv2.drawContours(np.zeros_like(land_mask_binary), [border_cnt_approx], -1, 255, -1)
    border_land_mask_approx = cv2.drawContours(border_land_mask_approx, [border_cnt_approx], -1, 0, 0)
    cv2.imwrite("border_land_mask_approx.png", border_land_mask_approx)

    land_mask_color = cv2.cvtColor(land_mask_binary, cv2.COLOR_GRAY2BGR)
    border_contours_img = cv2.drawContours(copy.deepcopy(land_mask_color), [border_cnt_approx], -1, (0,255,0), 2)
    cv2.imwrite("border_contours.png", border_contours_img)

    #approximate island contours
    island_cnts_approx = []
    for i, cnt in enumerate(island_contours):
        eps = epsilon * cv2.arcLength(cnt, True)
        cnt_approx = cv2.approxPolyDP(cnt, eps, True)
        cvx_hull = cv2.convexHull(cnt_approx)
        if len(cvx_hull) > 1:
            island_cnts_approx.append(cvx_hull)
            print(len(cnt), "->", len(cvx_hull), "vertices")
    print()

    island_contours_img = cv2.drawContours(copy.deepcopy(land_mask_color), island_cnts_approx, -1, (0,255,0), 2)
    cv2.imwrite("island_contours.png", island_contours_img)

    #convex island masks
    island_mask_approx = cv2.drawContours(255*np.ones_like(island_binary), island_cnts_approx, -1, 0, -1)
    cv2.imwrite("island_mask_approx.png", island_mask_approx)

    #final approximate land mask
    land_mask_approx = border_land_mask_approx/255 * island_mask_approx/255
    print(land_mask_approx.dtype)
    cv2.imwrite("land_mask_approx.png", 255*land_mask_approx)

    # obstacles = []
    # for p in cnts_approx

    # poly_points = np.concatenate(cnts_approx[4], axis=0)
    # poly_points[:, 1] = -(poly_points[:, 1] - (c.shape[0] - 1))
    # # print(min(poly_points[:, 0]))
    # # print(max(poly_points[:, 0]))
    # # print(min(poly_points[:, 1])) 
    # # print(max(poly_points[:, 1]))
    # s_poly = shapely.Polygon(poly_points)
    # line = shapely.LineString([(500, 0), (550, 600)])

    # start = timer()
    # intersec = shapely.intersection(s_poly, line)
    # end = timer()
    # int_compute_time = end - start
    # n_rays = 16
    # n_agents = 6
    # worst_case_total_rc_compute = int_compute_time * len(cnts_approx) * n_rays * n_agents
    # print("Intersection compute time:", end - start)
    # print("Worse-case ray casting compute time:", worst_case_total_rc_compute)
    # print(intersec)