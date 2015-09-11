import time
import numpy as np
from bokeh.plotting import figure, save, output_file 
from bokeh.models import ColumnDataSource
from bokeh.palettes import Greys9
import h5py

def convert_rgb_to_bokehrgba(img):
    """
    convert RGB image to two-dimensional array of RGBA values (encoded as 32-bit integers)

    Bokeh require rbga
    :param img: (N,M, 3) array (dtype = uint8)
    :return: (K, R, dtype=uint32) array
    """
    if img.dtype != np.uint8:
        raise NotImplementedError

    if img.ndim != 3:
        raise NotImplementedError

    bokeh_img = np.dstack([img, 255 * np.ones(img.shape[:2], np.uint8)])
    final_rgba_image = np.squeeze(bokeh_img.view(dtype=np.uint32))
    return final_rgba_image

f = h5py.File('./old_data.h5', 'r')

dh = 32
dw = 32

rows = list()
rowfigs = list()
rowlength = 6
numfigs = 96

for i in range(numfigs):
    imgdata = f['deconv/layer1/feature_map'+str(i)][...]

    imgdata = (0xff*np.transpose(imgdata, (1,2,0))).astype(np.uint8)

    
    p = figure(title="test image" + str(i), title_text_font_size='8pt',
            x_range=[0,dw],
            y_range=[0,dh], plot_width=150, plot_height=150, toolbar_location=None)

    p.axis.visible = None

    # Must multiply by 0xff to ensure they don't all end up rounding down to 0
    bokeh_img = np.dstack([imgdata, 255 * np.ones(imgdata.shape[:2], np.uint8)])
    final_image = bokeh_img.reshape(32*32*4).view(np.uint32).reshape((32,32))
    p.image_rgba([final_image], x = [0], y = [0], dw=[dw], dh=[dh])
    p.min_border = 0
    p.h_symmetry = False
    if len(rowfigs) < rowlength:
        rowfigs.append(p)
        #print(len(rowfigs))
    else:
        rows.append(hplot(*rowfigs))
        rowfigs = list()
    

if len(rowfigs):
    rows.append(hplot(*rowfigs))

allfigs = vplot(*rows)

print(allfigs)
output_file("image.html")

save(allfigs)
