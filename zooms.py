#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:56:05 2021

@author: mckay

Create Mandelbrot zooms or paths along Bezier curves
"""

import os
import numpy as np
import re
import imageio
from scipy.interpolate import interp1d
# from scipy.special import comb

from mandelbrot import MandelbrotMakerGPU

# def bernstein_poly(k, n, t):
#     return comb(n, k) * t**k * (1-t)**(n-k)

# def bezier_curve(x, y, resolution=100):
#     n_points = len(x)
#     assert len(y) == n_points, 'x and y should have the same length'
#     t = np.linspace(0, 1, resolution)
#     poly_array = np.array([bernstein_poly(k, n_points-1, t) \
#                           for k in range(n_points)])
#     xout = np.dot(x, poly_array)
#     yout = np.dot(y, poly_array)
#     return xout, yout

def smooth_curve(x, y, resolution=100):
    """Uses cubic splines to create a smooth path in 2d"""
    assert len(x) == len(y), 'x and y should have the same length'
    t_ = np.linspace(0, 1, len(x))
    t = np.linspace(0, 1, resolution)
    xout = interp1d(t_, x, kind='cubic')(t)
    yout = interp1d(t_, y, kind='cubic')(t)
    return xout, yout

# # This whole thing doesn't work. It creates artifacts from stretching... :(
# # I need to be able to do sub-pixel shifting if I want this to work
# def recycle_frame(last_frame, last_xstart, last_xstop, last_ystart, last_ystop,
#                   xstart, xstop, ystart, ystop, mm):
#     last_width = last_xstop - last_xstart
#     last_height = last_ystop - last_ystart
#     if xstart > last_xstart:
#         xmove = 0  # move right
#         lost_width = xstart - last_xstart
#         lost_pixelwidth = int(last_frame.shape[1] * lost_width / last_width)
#         keep = last_frame[:, lost_pixelwidth:]
#         vert_xstart, vert_xstop = last_xstop, xstop
#     else:
#         xmove = 1  # move right
#         lost_width = last_xstart - xstart
#         lost_pixelwidth = int(last_frame.shape[1] * lost_width / last_width)
#         keep = last_frame[:, :-lost_pixelwidth]
#         vert_xstart, vert_xstop = xstart, last_xstart
#     if ystart > last_ystart:
#         ymove = 0  # move down
#         lost_height = ystart - last_ystart
#         lost_pixelheight = int(last_frame.shape[1] * lost_height / last_height)
#         keep = keep[lost_pixelheight:, :]
#         horiz_ystart, horiz_ystop = last_ystop, ystop
#     else:
#         ymove = 1  # move up
#         lost_height = last_ystart - ystart
#         lost_pixelheight = int(last_frame.shape[1] * lost_height / last_height)
#         keep = keep[:-lost_pixelheight, ]
#         horiz_ystart, horiz_ystop = ystart, last_ystart
#     assert lost_pixelwidth >= 0
#     assert lost_pixelheight >= 0
#     # assemble new vertical component
#     pxwidth = last_frame.shape[1] - keep.shape[1]
#     if pxwidth > 0:
#         # create vertical bar that goes to right or left of kept section
#         vert = mm.gen_mandelbrot(vert_xstart, vert_xstop,
#                                  ystart, ystop,
#                                  pixelwidth=pxwidth,
#                                  pixelheight=last_frame.shape[0])
#     else:
#         vert = None
#     # assemble new horizontal component
#     pxheight = last_frame.shape[0] - keep.shape[0]
#     if pxheight > 0:
#         if xmove == 0:
#             horiz = mm.gen_mandelbrot(xstart, vert_xstart,
#                                       horiz_ystart, horiz_ystop,
#                                       pixelwidth=keep.shape[1],
#                                       pixelheight=pxheight)
#         else:
#             horiz = mm.gen_mandelbrot(vert_xstop, xstop,
#                                       horiz_ystart, horiz_ystop,
#                                       pixelwidth=keep.shape[1],
#                                       pixelheight=pxheight)
#     else:
#         horiz = None
#     # now put everything together
#     if vert is not None and horiz is not None:
#         if xmove == 0 and ymove == 1:
#             f = np.concatenate((np.concatenate((horiz, keep), 0), vert), 1)
#         elif xmove == 0 and ymove == 0:
#             f = np.concatenate((np.concatenate((keep, horiz), 0), vert), 1)
#         elif xmove == 1 and ymove == 1:
#             f = np.concatenate((vert, np.concatenate((horiz, keep), 0)), 1)
#         else:
#             f = np.concatenate((vert, np.concatenate((keep, horiz), 0)), 1)
#     # Deal with cases where we don't need one of the slices
#     elif vert is None and horiz is None:
#         f = keep
#     elif vert is None:
#         if ymove == 0:
#             f = np.concatenate((keep, horiz), 0)
#         else:
#             f = np.concatenate((horiz, keep), 0)
#     else:
#         if xmove == 0:
#             f = np.concatenate((keep, vert), 1)
#         else:
#             f = np.concatenate((vert, keep), 1)
#     return f


def gen_path_from_coords(pixelwidth=512, pixelheight=None, nframes=100,
                         coord_source='coordinates.txt',
                         save_loc='~/Documents/python/', keep_frames=False,
                         quality=10, fps=30, buffer_frames=30, colors=None):
    # import and process the list of coordinates
    save_loc = os.path.expanduser(save_loc)
    if not os.path.isdir(save_loc):
        os.mkdir(save_loc)
    with open(coord_source, 'r') as fh:
        coords = fh.read().strip().split('\n')
    searches = [re.search(r'^re (-?[\d\.]+)_(-?[\d\.]+) im (-?[\d\.]+)_(-?[\d\.]+)$', x) \
                for x in coords]
    xstarts = np.array([float(s.group(1)) for s in searches])
    xstops = np.array([float(s.group(2)) for s in searches])
    ystarts = np.array([float(s.group(3)) for s in searches])
    ystops = np.array([float(s.group(4)) for s in searches])
    xstart_path, ystart_path = smooth_curve(xstarts, ystarts, nframes)
    xstop_path, ystop_path = smooth_curve(xstops, ystops, nframes)
    if pixelheight is None:
        pixelheight = int(pixelwidth * (ystops[0] - ystarts[0]) / (xstops[0] - xstarts[0]))
    if colors is None:
        mm = MandelbrotMakerGPU()
    else:
        mm = MandelbrotMakerGPU(c0=colors[0], c1=colors[1], c2=colors[2],
                                c3=colors[3], c4=colors[4])
    frames = []
    for i in range(0, nframes):
        f= mm.gen_mandelbrot(xstart_path[i], xstop_path[i],
                             ystart_path[i], ystop_path[i],
                             pixelwidth=pixelwidth, pixelheight=pixelheight)
        # save frame
        if keep_frames:
            if not os.path.isdir(os.path.join(save_loc, 'frames')):
                os.mkdir(os.path.join(save_loc, 'frames'))
            imageio.imwrite(uri=os.path.join(save_loc, 'frames', f'frame_{i}.png'),
                            im=f)
        frames.append(f)
        if i % 10 == 9:
            print(f'Completed {i+1} of {nframes} frames')
    frames = [frames[0]]*buffer_frames + frames + [frames[-1]]*buffer_frames
    imageio.mimwrite(os.path.join(save_loc, 'bezier_path.mp4'),
                     frames, fps=fps, quality=quality)
