import contextily as cx
import copy
import cv2
import mercantile as mt
import numpy as np

from geographiclib.geodesic import Geodesic
from math import ceil
from PIL import Image
from scipy.ndimage import label


COORD1 = (42.306671732900284, -71.0306793176816)
COORD2 = (42.35077795666068, -70.94882396266765)

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
        w=COORD1[1], s=COORD1[0], e=COORD2[1], n=COORD2[0], zoom="auto", source=source, ll=True,
        wait=0, max_retries=2, n_connections=1, use_cache=False, zoom_adjust=None
    )
    print("Initial image size:", img[:,:,:-1].shape)

    img, ext = crop_tiles(img[:,:,:-1], ext, COORD1[1], COORD1[0], COORD2[1], COORD2[0], ll=True)
    print("Final image size:", img.shape)

    image = Image.fromarray(img)
    image.save("topo_test.png")

    # cv2_topo_img = cv2.imread("topo_test.png")
    # cv2_topo_img = cv2.cvtColor(cv2_topo_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("inspect", cv2_topo_img)
    # cv2.waitKey(0)

    # test distance between mercator and gps distance (3 miles)
    water = (42.32418681048927, -71.02230485803544)
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
    labeled_mask, n_objects = label(mask, structure=water_connectivity)
    target_label = labeled_mask[water_pixel_y, water_pixel_x]
    water_mask = labeled_mask == target_label

    water_image = Image.fromarray(water_mask)
    water_image.save("water_mask.png")

    water_binary = 255*water_mask.astype(np.uint8)
    land_binary = 255*np.logical_not(water_mask).astype(np.uint8)

    #contours (https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?#findcontours)
    water_contours, wh = cv2.findContours(water_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    land_contours, lh = cv2.findContours(land_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    water_color_img = cv2.cvtColor(water_binary, cv2.COLOR_GRAY2BGR)

    a = cv2.drawContours(copy.deepcopy(water_color_img), water_contours, -1, (0,255,0), 3)
    b = cv2.drawContours(copy.deepcopy(water_color_img), land_contours, -1, (0,255,0), 3)

    cv2.imwrite("water_contours.png", a)
    cv2.imwrite("land_contours.png", b)

    #approximate contours
    epsilon = 0.0005

    water_cnt = max(water_contours, key=cv2.contourArea)
    land_cnt = max(land_contours, key=cv2.contourArea)

    water_epsilon = epsilon * cv2.arcLength(water_cnt,True)
    land_epsilon = epsilon * cv2.arcLength(land_cnt,True)

    water_approx = cv2.approxPolyDP(water_cnt, water_epsilon, True)
    land_approx = cv2.approxPolyDP(land_cnt, land_epsilon, True)

    print(water_cnt.shape)
    print(water_approx.shape)

    c = cv2.drawContours(copy.deepcopy(water_color_img), [water_approx], -1, (0,255,0), 3)
    d = cv2.drawContours(copy.deepcopy(water_color_img), [land_approx], -1, (0,255,0), 3)

    cv2.imwrite("approx_water_contour.png", c)
    cv2.imwrite("approx_land_contour.png", d)

    # Geodesic.WGS84.Direct(lat1=34., lon1=148., azi1=90., s12=10_000.) #s12 is the distance from the first point to the second in meters