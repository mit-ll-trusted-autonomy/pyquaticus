import contextily as cx
import copy
import cv2
import mercantile as mt
import numpy as np
import shapely

from geographiclib.geodesic import Geodesic
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
        w=COORD1[1], s=COORD1[0], e=COORD2[1], n=COORD2[0], zoom=12, source=source, ll=True,
        wait=0, max_retries=2, n_connections=1, use_cache=False, zoom_adjust=None
    )
    print("Initial image size:", img[:,:,:-1].shape)

    img, ext = crop_tiles(img[:,:,:-1], ext, COORD1[1], COORD1[0], COORD2[1], COORD2[0], ll=True)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("Final image size:", img.shape)

    image = Image.fromarray(img)
    image.save("topo_test.png")

    # cv2_topo_img = cv2.imread("topo_test.png")
    # cv2_topo_img = cv2.cvtColor(cv2_topo_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("inspect", cv2_topo_img)
    # cv2.waitKey(0)

    # test distance between mercator and gps distance (3 miles)
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
    water_mask = (labeled_mask == target_label) + (38 <= gray_img) * (gray_img <= 40)
    land_mask = np.logical_not(water_mask)

    water_image = Image.fromarray(water_mask)
    water_image.save("water_mask.png")

    water_binary = 255*water_mask.astype(np.uint8)
    land_binary = 255*land_mask.astype(np.uint8)

    #contours (https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?#findcontours)
    land_contours_old, _ = cv2.findContours(land_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    water_color_img = cv2.cvtColor(water_binary, cv2.COLOR_GRAY2BGR)
    land_color_img = cv2.cvtColor(land_binary, cv2.COLOR_GRAY2BGR)

    water_contours_img = cv2.drawContours(copy.deepcopy(water_color_img), land_contours_old, -1, (0,255,0), 3)
    land_contours_img = cv2.drawContours(copy.deepcopy(land_color_img), land_contours_old, -1, (0,255,0), 3)

    cv2.imwrite("water_contours_old.png", water_contours_img)

    #remove holes inside land (water) with water contours
    land_contours_img_gray = cv2.cvtColor(land_contours_img, cv2.COLOR_BGR2GRAY)
    water_pixel_color_gray = land_contours_img_gray[water_pixel_y, water_pixel_x]
    gray_water_mask = land_contours_img_gray == water_pixel_color_gray
    labeled_mask_gray, _ = label(gray_water_mask, structure=water_connectivity)
    target_label_gray = labeled_mask_gray[water_pixel_y, water_pixel_x]
    land_mask_new = np.logical_not(labeled_mask_gray == target_label_gray)

    #contours 2
    land_binary_new = 255*land_mask_new.astype(np.uint8)

    land_contours_new, _ = cv2.findContours(land_binary_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print("# land contours old:", len(land_contours_old))
    print("# land contours new:", len(land_contours_new))

    land_contours_img_new = cv2.drawContours(copy.deepcopy(water_color_img), land_contours_new, -1, (0,255,0), 3)

    cv2.imwrite("water_contours_new.png", land_contours_img_new)

    #approximate contours
    epsilon = 0.001
    cnts_approx = []

    for i, cnt in enumerate(land_contours_new):
        eps = epsilon * cv2.arcLength(cnt,True)
        cnt_approx = cv2.approxPolyDP(cnt, eps, True)
        cnts_approx.append(cnt_approx)
        print(len(cnt), "->", len(cnt_approx), "vertices")

    c = cv2.drawContours(copy.deepcopy(water_color_img), cnts_approx, -1, (0,255,0), 3)
    cv2.imwrite("approx_water_contour.png", c)

    poly_points = np.concatenate(cnts_approx[4], axis=0)
    poly_points[:, 1] = -(poly_points[:, 1] - (c.shape[0] - 1))
    # print(min(poly_points[:, 0]))
    # print(max(poly_points[:, 0]))
    # print(min(poly_points[:, 1])) 
    # print(max(poly_points[:, 1]))
    s_poly = shapely.Polygon(poly_points)
    line = shapely.LineString([(500, 0), (550, 600)])
    
    start = timer()
    intersec = shapely.intersection(s_poly, line)
    end = timer()
    int_compute_time = end - start
    n_rays = 16
    n_agents = 6
    worst_case_total_rc_compute = int_compute_time * len(cnts_approx) * n_rays * n_agents
    print("Intersection compute time:", end - start)
    print("Worse-case ray casting compute time:", worst_case_total_rc_compute)
    print(intersec)

    # Geodesic.WGS84.Direct(lat1=34., lon1=148., azi1=90., s12=10_000.) #s12 is the distance from the first point to the second in meters