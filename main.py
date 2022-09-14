#!/usr/bin/env python3

from math import sin, pi
from cmath import rect
from numpy import linspace, array, packbits, bitwise_and, invert, shape
from PIL import Image, ImageDraw
from random import uniform, randint, randrange

import os
from multiprocessing import Pool
import time

resolution = 10000

def noise(theta, weights, phases):
    # https://gamedev.stackexchange.com/questions/62613/need-ideas-for-an-algorithm-to-draw-irregular-blotchy-shapes
    return 1 + 0.25*(weights[0]*sin(theta+(2*pi)*phases[0])) \
        + weights[1]*sin(2*theta+2*pi*phases[1]) \
        + weights[2]*sin(3*theta+2*pi*phases[2]) \
        + weights[3]*sin(4*theta+2*pi*phases[3]) \
        + weights[4]*sin(5*theta+2*pi*phases[4])

def generate_parasite(output_path, radius_scale, offset):
    image = Image.new("1", (resolution, resolution), "white")
    draw = ImageDraw.Draw(image)

    phases = [uniform(0, 1) for i in range(5)]
    weights = [uniform(0, 1)/i for i in range(1, 6)] # brownian noise

    x = []
    y = []
    angles = linspace(-16, 16, num=400)
    for i in angles:
        r = radius_scale*noise(i, weights, phases)
        rect_coord = rect(r, i)
        x.append(rect_coord.real + radius_scale + offset)
        y.append(-1*rect_coord.imag + radius_scale + offset)

    draw.line(list(zip(x, y)), fill=0, width=2)
    ImageDraw.floodfill(image, (radius_scale+offset, radius_scale+offset), 0)
    image.save(output_path)

def generate_dye(output_path, radius_scale, offset):
    image = Image.new("1", (resolution, resolution), "black")
    draw = ImageDraw.Draw(image)

    num_dye_spots = randint(0, 100)
    for i in range(num_dye_spots):
        radius = randrange(1, radius_scale//100)
        center_x = randrange(offset, 2*radius_scale + offset)
        center_y = randrange(offset, 2*radius_scale + offset)
        draw.ellipse(xy=(center_x - radius,
                         center_y - radius,
                         center_x + radius,
                         center_y + radius),
                     fill=1)

    image.save(output_path)

def has_cancer(total_parasite_size, total_overlap):
    return (total_overlap/total_parasite_size) > 0.1

def analyse_image(parasite_path, dye_path):
    parasite = packbits(array(Image.open(parasite_path).convert("1")))
    dye = packbits(array(Image.open(dye_path).convert("1")))
    total_parasite_size = 0
    total_overlap = 0
    start_time = time.time()
    for i in range(len(parasite)):
        total_parasite_size += bin(invert(parasite[i])).count('1')
        total_overlap += bin(bitwise_and(invert(parasite[i]), dye[i])).count('1')
    print("--- {time} | Single thread ---".format(time=time.time()-start_time))
    print(" Total parasite size: {total_parasite_size} | Total overlap size {total_overlap}".format(total_parasite_size=total_parasite_size, total_overlap=total_overlap))
    return has_cancer(total_parasite_size, total_overlap)

def analyse_parasite(parasite, dye):
    parasite_size = bin(invert(parasite)).count('1') # count number of parasite pixels inverted
    overlap = bin(bitwise_and(invert(parasite), dye)).count('1') # count inverted parasite and dye pixels
    return (parasite_size, overlap)

def analyse_image_multithreaded(parasite_path, dye_path):
    parasite = packbits(array(Image.open(parasite_path).convert("1")))
    dye = packbits(array(Image.open(dye_path).convert("1")))

    total_parasite_size = 0
    total_overlap = 0
    nproc = os.cpu_count()
    start_time = time.time()
    with Pool(processes=nproc) as pool:
        for bact, overlap in pool.starmap(analyse_parasite, list(zip(parasite, dye)), chunksize=shape(parasite)[0]//nproc):
            total_parasite_size += bact
            total_overlap += overlap
    print("--- {time} | Multithread {chunksize} ---".format(time=time.time()-start_time, chunksize=shape(parasite)[0]//nproc))
    print(" Total parasite size: {total_parasite_size} | Total overlap size {total_overlap}".format(total_parasite_size=total_parasite_size, total_overlap=total_overlap))
    return has_cancer(total_parasite_size, total_overlap)

if __name__ == "__main__":

    num_test_images = 5

    # generate test data
    for i in range(num_test_images):
        radius_scale = randint(resolution//4, resolution//2) # parasite must take up at least 25% of total
        offset = randint(0, radius_scale//2) # additional offset for placement on image
        generate_parasite("parasite-{index}.pgm".format(index = i), radius_scale, offset)
        generate_dye("dye-{index}.pgm".format(index = i), radius_scale, offset)

    # test parasite image for cancer
    for i in range(num_test_images):
        result = analyse_image("parasite-{index}.jpg".format(index = i), "dye-{index}.jpg".format(index = i))
        # result = analyse_image_multithreaded("parasite-{index}.jpg".format(index = i), "dye-{index}.jpg".format(index = i))
        if result:
          print("parasite {index} has cancer".format(index=i))
        else:
          print("parasite {index} does not have cancer".format(index=i))
