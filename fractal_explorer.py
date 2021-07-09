#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:13:18 2021

@author: mckay
"""

import sys
import base64
import PySimpleGUI as sg
import numpy as np
from PIL import Image
from io import BytesIO

from mandelbrot import MandelbrotMaker, MandelbrotMakerGPU

# Default start coordinates
XSTART = -64/27
XSTOP = 32/27
YSTART = -1
YSTOP = 1

### Engine for moving around in the fractal space

class ImageEngine:
    
    def __init__(self, pixelwidth, pixelheight=0, use_gpu=False,
                 xstart=XSTART, xstop=XSTOP, ystart=YSTART, ystop=YSTOP, mm=None):
        self.pixelwidth = pixelwidth
        self.pixelheight = pixelheight
        self.use_gpu = use_gpu
        self.xstart = xstart
        self.xstop = xstop
        self.ystart = ystart
        self.ystop = ystop
        if mm is None:
            self.mm = MandelbrotMakerGPU() if use_gpu else MandelbrotMaker()
        else:
            self.mm = mm
        
    
    def gen_grid(self):
        return self.mm.gen_mandelbrot(self.xstart, self.xstop,
                                      self.ystart, self.ystop,
                                      pixelwidth=self.pixelwidth,
                                      pixelheight=self.pixelheight)
    
    def get_image_str(self, regen=True):
        if regen:
            self.grid = self.gen_grid()
        buffered = BytesIO()
        Image.fromarray(self.grid).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str
    
    def move_up(self):
        delta = self.ystop - self.ystart
        new_ystart = self.ystart - 0.25 * delta
        helper = ImageEngine(self.pixelwidth, self.pixelheight,
                             self.use_gpu,
                             self.xstart, self.xstop,
                             new_ystart, self.ystart, mm=self.mm)
        helper_grid = helper.gen_grid()
        stop_row = -helper_grid.shape[0]
        self.grid = np.concatenate((helper_grid, self.grid[:stop_row, :]), axis=0)
        self.ystart = new_ystart
        self.ystop -= 0.25 * delta
        return self.get_image_str(regen=False)
    
    def move_down(self):
        delta = self.ystop - self.ystart
        new_ystop = self.ystop + 0.25 * delta
        helper = ImageEngine(self.pixelwidth, self.pixelheight,
                             self.use_gpu,
                             self.xstart, self.xstop,
                             self.ystop, new_ystop, mm=self.mm)
        helper_grid = helper.gen_grid()
        start_row = helper_grid.shape[0]
        self.grid = np.concatenate((self.grid[start_row:, :], helper_grid), axis=0)
        self.ystop = new_ystop
        self.ystart += 0.25 * delta
        return self.get_image_str(regen=False)
    
    def move_left(self):
        delta = self.xstop - self.xstart
        new_xstart = self.xstart - 0.25 * delta
        pxh = int(self.pixelwidth * (self.ystop - self.ystart)
                  / (self.xstop - self.xstart)) \
            if self.pixelheight == 0 else self.pixelheight
        helper = ImageEngine(self.pixelwidth // 4, pxh,
                             self.use_gpu,
                             new_xstart, self.xstart,
                             self.ystart, self.ystop, mm=self.mm)
        helper_grid = helper.gen_grid()
        stop_col = -helper_grid.shape[1]
        self.grid = np.concatenate((helper_grid, self.grid[:, :stop_col]), axis=1)
        self.xstart = new_xstart
        self.xstop -= 0.25 * delta
        return self.get_image_str(regen=False)
    
    def move_right(self):
        delta = self.xstop - self.xstart
        new_xstop = self.xstop + 0.25 * delta
        pxh = int(self.pixelwidth * (self.ystop - self.ystart)
                  / (self.xstop - self.xstart)) \
            if self.pixelheight == 0 else self.pixelheight
        helper = ImageEngine(self.pixelwidth // 4, pxh,
                             self.use_gpu,
                             self.xstop, new_xstop,
                             self.ystart, self.ystop, mm=self.mm)
        helper_grid = helper.gen_grid()
        start_col = helper_grid.shape[1]
        self.grid = np.concatenate((self.grid[:, start_col:], helper_grid), axis=1)
        self.xstop = new_xstop
        self.xstart += 0.25 * delta
        return self.get_image_str(regen=False)
    
    def zoom_in(self):
        if (self.xstop - self.xstart) < 1e-13:
            # Reached zoom limit due to floating point precision
            return self.get_image_str(regen=False)
        xmedian = (self.xstop + self.xstart) / 2
        ymedian = (self.ystop + self.ystart) / 2
        self.xstop -= (0.75 * (self.xstop - xmedian))
        self.xstart += (0.75 * (xmedian - self.xstart))
        self.ystop -= (0.75 * (self.ystop - ymedian))
        self.ystart += (0.75 * (ymedian - self.ystart))
        return self.get_image_str()
    
    def zoom_out(self):
        xmedian = (self.xstop + self.xstart) / 2
        ymedian = (self.ystop + self.ystart) / 2
        self.xstop += (3 * (self.xstop - xmedian))
        self.xstart -= (3 * (xmedian - self.xstart))
        self.ystop += (3 * (self.ystop - ymedian))
        self.ystart -= (3 * (ymedian - self.ystart))
        return self.get_image_str()
    
    def get_coords(self):
        return f're {self.xstart}_{self.xstop} ' \
            + f'im {self.ystart}_{self.ystop}'
            

### main function controls user interface

def main(pixelwidth=1280, use_gpu=False):
    imageEngine = ImageEngine(pixelwidth=pixelwidth, use_gpu=use_gpu)
    image = sg.Image(data=imageEngine.get_image_str())
    layout = [
            [sg.Text('Color 0:'), sg.Input(str(imageEngine.mm.c0), key='-COLOR0-')],
            [sg.Text('Color 1:'), sg.Input(str(imageEngine.mm.c1), key='-COLOR1-')],
            [sg.Text('Color 2:'), sg.Input(str(imageEngine.mm.c2), key='-COLOR2-')],
            [sg.Text('Color 3:'), sg.Input(str(imageEngine.mm.c3), key='-COLOR3-')],
            [sg.Text('Color 4:'), sg.Input(str(imageEngine.mm.c4), key='-COLOR4-')],
            [sg.Button('Update colors')],
            [image],
            [sg.Button('Up')],
            [sg.Button('Left'), sg.Button('Right')],
            [sg.Button('Down')],
            [sg.Button('-'), sg.Button('+')],
            [sg.Button('Save coordinates'), sg.Button('Save'), sg.Button('Reset')]
        ]
    window = sg.Window('Fractal Explorer', layout, element_justification='c')
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Up':
            image.update(data=imageEngine.move_up())
        elif event == 'Down':
            image.update(data=imageEngine.move_down())
        elif event == 'Left':
            image.update(data=imageEngine.move_left())
        elif event == 'Right':
            image.update(data=imageEngine.move_right())
        elif event == '+':
            image.update(data=imageEngine.zoom_in())
        elif event == '-':
            image.update(data=imageEngine.zoom_out())
        elif event == 'Update colors':
            try:
                c0 = tuple(map(int, values['-COLOR0-'].strip('()').split(',')))
            except ValueError:
                pass
            if c0 != imageEngine.mm.c0:
                imageEngine.mm.c0 = c0
                image.update(data=imageEngine.get_image_str())
            window['-COLOR0-'].update(str(imageEngine.mm.c0))
            try:
                c1 = tuple(map(int, values['-COLOR1-'].strip('()').split(',')))
            except ValueError:
                pass
            if c1 != imageEngine.mm.c1:
                imageEngine.mm.c1 = c1
                image.update(data=imageEngine.get_image_str())
            window['-COLOR1-'].update(str(imageEngine.mm.c1))
            try:
                c2 = tuple(map(int, values['-COLOR2-'].strip('()').split(',')))
            except ValueError:
                pass
            if c2 != imageEngine.mm.c2:
                imageEngine.mm.c2 = c2
                image.update(data=imageEngine.get_image_str())
            window['-COLOR2-'].update(str(imageEngine.mm.c2))
            try:
                c3 = tuple(map(int, values['-COLOR3-'].strip('()').split(',')))
            except ValueError:
                pass
            if c3 != imageEngine.mm.c3:
                imageEngine.mm.c3 = c3
                image.update(data=imageEngine.get_image_str())
            window['-COLOR3-'].update(str(imageEngine.mm.c3))
            try:
                c4 = tuple(map(int, values['-COLOR4-'].strip('()').split(',')))
            except ValueError:
                pass
            if c4 != imageEngine.mm.c4:
                imageEngine.mm.c4 = c4
                image.update(data=imageEngine.get_image_str())
            window['-COLOR4-'].update(str(imageEngine.mm.c4))
        elif event == 'Save coordinates':
            with open('coordinates.txt', 'a') as fh:
                fh.write(imageEngine.get_coords() + '\n')
        elif event == 'Save image':
            Image.fromarray(imageEngine.grid).save(
                    imageEngine.get_coords() + '.png'
                )
        elif event == 'Reset':
            imageEngine.xstart = XSTART
            imageEngine.xstop = XSTOP
            imageEngine.ystart = YSTART
            imageEngine.ystop = YSTOP
            image.update(data=imageEngine.get_image_str())
    window.close()


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        try:
            assert args[2] in ('0', '1')
            pixelwidth = int(args[1])
            use_gpu = bool(int(args[2]))
        except:
            print('Syntax is "python fractal_explorer.py [pixelwidth] [use_gpu (0/1)]"')
            raise
        main(pixelwidth, use_gpu)
    else:
        main()
        
