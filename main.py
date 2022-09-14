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

def generate_bacteria(output_path, radius_scale, offset):
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

def has_cancer(total_bacteria_size, total_overlap):
    return (total_overlap/total_bacteria_size) > 0.1

def analyse_image(bacteria_path, dye_path):
    bacteria = packbits(array(Image.open(bacteria_path).convert("1")))
    dye = packbits(array(Image.open(dye_path).convert("1")))
    total_bacteria_size = 0
    total_overlap = 0
    start_time = time.time()
    for i in range(len(bacteria)):
        total_bacteria_size += bin(invert(bacteria[i])).count('1')
        total_overlap += bin(bitwise_and(invert(bacteria[i]), dye[i])).count('1')
    print("--- {time} | Single thread ---".format(time=time.time()-start_time))
    return has_cancer(total_bacteria_size, total_overlap)

def analyse_bacteria(bacteria, dye):
    bacteria_size = bin(invert(bacteria)).count('1') # count number of bacteria pixels inverted
    overlap = bin(bitwise_and(invert(bacteria), dye)).count('1') # count inverted bacteria and dye pixels
    return (bacteria_size, overlap)

def analyse_image_multithreaded(bacteria_path, dye_path):
    bacteria = packbits(array(Image.open(bacteria_path).convert("1")))
    dye = packbits(array(Image.open(dye_path).convert("1")))

    total_bacteria_size = 0
    total_overlap = 0
    nproc = os.cpu_count()
    start_time = time.time()
    with Pool(processes=nproc) as pool:
        for bact, overlap in pool.starmap(analyse_bacteria, list(zip(bacteria, dye)), chunksize=shape(bacteria)[0]//nproc):
            total_bacteria_size += bact
            total_overlap += overlap
    print("--- {time} | Multithread {chunksize} ---".format(time=time.time()-start_time, chunksize=shape(bacteria)[0]//nproc))
    return has_cancer(total_bacteria_size, total_overlap)

if __name__ == "__main__":

    num_test_images = 5

    # generate test data
    for i in range(num_test_images):
        radius_scale = randint(resolution//4, resolution//2) # bacteria must take up at least 25% of total
        offset = randint(0, radius_scale//2) # additional offset for placement on image
        generate_bacteria("bacteria-{index}.jpg".format(index = i), radius_scale, offset)
        generate_dye("dye-{index}.jpg".format(index = i), radius_scale, offset)

    # test parasite image for cancer
    for i in range(num_test_images):
        result = analyse_image("bacteria-{index}.jpg".format(index = i), "dye-{index}.jpg".format(index = i))
        # result = analyse_image_multithreaded("bacteria-{index}.jpg".format(index = i), "dye-{index}.jpg".format(index = i))
        if result:
          print("bacteria {index} has cancer".format(index=i))
        else:
          print("bacteria {index} does not have cancer".format(index=i))
