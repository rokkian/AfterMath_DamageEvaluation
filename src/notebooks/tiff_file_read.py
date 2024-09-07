#%%
import rasterio
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from rasterio.windows import Window
import multiprocessing as mp
from icecream import ic
from joblib import Parallel, delayed

os.chdir("/Users/mrokk/workspace/tlHack/AfterMath_DamageEvaluation/")

#%%
with rasterio.open("../Luch_ortophoto.tif") as dataset:
    # Read the image data, define a window for partial reading
    image_data_r = dataset.read(1)

#%%
image_data_r.shape # 59_724, 121904

#%% rows, cols
w = Window.from_slices((10000, 24096), (10000, 22048))

#%%
with rasterio.open("../Luch_ortophoto.tif") as dataset:
    # Read the image data, define a window for partial reading
    image_data_r = dataset.read(1, window=w)  # Assuming a single band image
    image_data_g = dataset.read(2, window=w)
    image_data_b = dataset.read(3, window=w)

#%%
with rasterio.open("../Luch_ortophoto.tif") as src:
    ic(src.width, src.height)
# %%
image_data_r, image_data_g, image_data_b

#%%
image_data = np.dstack([image_data_r, image_data_g, image_data_b])
# %%
image_data.shape
# %%
# np.max(image_data_g)
# %% show the image
plt.imshow(image_data)
plt.show()
# %%
def process_tile(tile_data):
    # Perform your ground detection or other processing on the tile data
    # ... 
    return tile_data

#%%
with rasterio.open('your_orthophoto.tif') as src:
    width = src.width
    height = src.height
    tile_size = 512

    with mp.Pool() as pool:
        results = []
        for i in range(0, width, tile_size):
            for j in range(0, height, tile_size):
                window = ((i, i + tile_size), (j, j + tile_size))
                tile = src.read(window=window)
                results.append(pool.apply_async(process_tile, (tile,)))

        processed_tiles = [result.get() for result in results]

# Combine or further process the 'processed_tiles' as needed
#%%
def read_tile(x0, y0, tile_size):
    results = []
    window = ((i, i + tile_size), (j, j + tile_size))
    with rasterio.open("../Luch_ortophoto.tif") as src:

    tile = src.read(window=window)
    results.appen
#%% con Joblib
Parallel(n_jobs=mp.cpu_count(), prefer="processes")(
    (delayed(process_tile)(x, y, tile_size=512) for x in range(0, width, tile_size) for y in range(0, height, tile_size))
)