# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from bokeh.plotting import figure, save, output_file
from bokeh.palettes import brewer
from bokeh.io import vplot, hplot
from bokeh.models import Range1d
import numpy as np

def x_label(epoch_axis):
    """
    Get the x axis label depending on the boolean epoch_axis.
    """
    return "Epoch" if epoch_axis else "Minibatch"


def cost_fig(cost_data, plot_height, plot_width, epoch_axis=True):
    """
    Generate a figure with lines for each element in cost_data.
    """
    fig = figure(plot_height=plot_height,
                 plot_width=plot_width,
                 title="Cost",
                 x_axis_label=x_label(epoch_axis),
                 y_axis_label="Cross Entropy Error (%)")

    # Spectral palette supports 3 - 11 distinct colors
    num_colors_required = len(cost_data)
    assert num_colors_required <= 11, "Insufficient colors in predefined palette."
    colors = brewer["Spectral"][max(3, len(cost_data))]
    if num_colors_required < 3:
        # manually adjust pallette for better contrast
        colors[0] = brewer["Spectral"][6][0]
        if num_colors_required == 2:
            colors[1] = brewer["Spectral"][6][-1]

    for name, x, y in cost_data:
        fig.line(x, y, legend=name, color=colors.pop(0), line_width=2)
    return fig

def deconv_map_to_rgb(img):
    minImg = np.min(img)
    img -= minImg
    maxImg = np.max(img)
    if maxImg == 0:
        maxImg = 1
    img = img / maxImg * 255
    return img

def convert_rgb_to_bokehrgba(imgdata, dh, dw):
    """
    convert RGB image to two-dimensional array of RGBA values (encoded as 32-bit integers)

    Bokeh require rbga
    :param img: (N,M, 3) array (dtype = uint8)
    :return: (K, R, dtype=uint32) array
    """
    if imgdata.dtype != np.uint8:
        raise NotImplementedError

    if imgdata.ndim != 3:
        raise NotImplementedError
            
    bokeh_img = np.dstack([imgdata, 255 * np.ones(imgdata.shape[:2], np.uint8)])
    # This step is necessary, because we transposed the data before passing it into this function
    # and somehow, that messed with some shape attribute. 
    final_image = bokeh_img.reshape(dh * dw *4).view(np.uint32)
    final_image = final_image.reshape((dh,dw))

    return final_image

def deconv_fig(img_data, plot_size, img_h, img_w, img_start=10, img_end=20, figs_per_row=10):
    """
    Generate a figure for each element in deconv_data.
    
    Arguments:
        img_data (tuple): feature map name, array with image rgb values
        plot_height (int): height of plot
        plot_width (int): width of plot
        img_h (int): pixel height of input image
        img_w (int): pixel width of input image
        img_start (int, optional): the x and y pixel range at which to start plotting
        img_end (int, optional): the x and y pixel range of the image at which to end  
        figs_per_row (int, optional): the number of images to plot in a row
    """
    rows = list()
    rowfigs = list()
    shared_range = Range1d(img_start, img_end)

    for fm_num, (fm_name, data) in enumerate(img_data):
        data = np.transpose(data, (1,2,0))
        rgb_img = deconv_map_to_rgb(data)
        rgb_img = rgb_img.astype(np.uint8)
        final_img = convert_rgb_to_bokehrgba(rgb_img, img_h, img_w)
        
        fig = figure(title=str(fm_num), title_text_font_size='6pt',
                    x_range=shared_range, y_range=shared_range,
                    plot_width=plot_size-15, plot_height = plot_size,
                    toolbar_location=None)
        fig.axis.visible = None

        fig.image_rgba([final_img], x=[0], y=[0], dw=[img_w], dh=[img_h])
        fig.min_border = 0

        if len(rowfigs) < figs_per_row:
            rowfigs.append(fig)
        else:
            rows.append(hplot(*rowfigs))
            rowfigs = list()

    if len(rowfigs):
        rows.append(hplot(*rowfigs))

    allfigs = vplot(*rows) 

    # TODO: take out
    output_file("image.html")
    save(allfigs)

    return allfigs

# TODO: e.g. figs = deconv_fig(ret, 65, 50, 32, 32, 15, 18, 10)
