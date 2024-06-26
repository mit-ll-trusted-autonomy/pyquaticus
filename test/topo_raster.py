import contextily as cx
import mercantile as mt
from geographiclib.geodesic import Geodesic
from PIL import Image


EPIC = (42.3495760, -71.1083459)
MIT_SAILING_PAVILION = (42.3585755, -71.0874269)

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

    crop_start_x = round(img_size_x * (left - ext[0]) / X_size)
    crop_end_x = round(img_size_x * (right - ext[0]) / X_size)

    crop_start_y = round(img_size_y * (ext[2] - top) / Y_size)
    crop_end_y = round(img_size_y * (ext[2] - bottom) / Y_size)

    #crop image
    image = img[
        crop_start_y : crop_end_y,
        crop_start_x : crop_end_x,
        :
    ]

    return image


if __name__ == "__main__":
    # source = cx.providers.CartoDB.VoyagerNoLabels
    source = cx.providers.OpenStreetMap.Mapnik
    img, ext = cx.bounds2img(
        w=EPIC[1], s=EPIC[0], e=MIT_SAILING_PAVILION[1], n=MIT_SAILING_PAVILION[0], zoom="auto", source=source, ll=True,
        wait=0, max_retries=2, n_connections=1, use_cache=False, zoom_adjust=None
    )
    print(img[:,:,:-1].shape)
    print()

    img = crop_tiles(img[:,:,:-1], ext, EPIC[1], EPIC[0], MIT_SAILING_PAVILION[1], MIT_SAILING_PAVILION[0], ll=True)
    print(img.shape)

    image = Image.fromarray(img)
    image.save("topo_test.png")

    # Geodesic.WGS84.Direct(lat1=34., lon1=148., azi1=90., s12=10_000.) #s12 is the distance from the first point to the second in meters