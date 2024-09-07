#%%
import numpy as np
import laspy
import os
import pyvista as pv
from pyvista import examples
from icecream import ic

os.chdir('/Users/mrokk/workspace/tlHack/AfterMath_DamageEvaluation/')

#%% loading the data
with laspy.open('../building-georeferenced_model.las') as fh:
    print('Points from Header:', fh.header.point_count)
    las = fh.read()
    print(las)
    print('Points from data:', len(las.points))
    ground_pts = las.classification == 2
    bins, counts = np.unique(las.return_number[ground_pts], return_counts=True)
    print('Ground Point Return Number distribution:')
    for r,c in zip(bins,counts):
        print('    {}:{}'.format(r,c))

# %% visualize the data
print(las)
print(las.point_format)
print("Pointcloud features:")
list(las.point_format.dimension_names)

# %%
points = np.column_stack((las.x, las.y, las.z))

# %% basic cloud plot
pv.plot(points)
# %%
pv.plot(points, scalars=np.column_stack((las.red, las.green, las.blue, las.intensity + 50)), rgba=True)

# %%
for dim in las.point_format.dimension_names:
    if dim in ("X", "Y", "Z"):
        continue
    ic(dim, np.unique(getattr(las, dim)), getattr(las, dim))
# %%
