# README

# Getting started

* Built with Python 3.10
* Linux environment (Gentoo)

```bash
pip install -r requirements.txt # ideally use virtualenv
python main.py
```

# Question 1

Ideally, both the parasite and dye images are represented as a 2D array of pixels where each pixel represents a single bit. However, the smallest indexable space in memory is a single byte. For example, Python will represent an array of boolean values as an array of uint8. The *bit packing* technique overcomes this challenge by each element in the array represents a number of pixels equal to the size of that element. For example, each element in an array of type unint8 represents 8 pixels. Similarly we can store the image on disk with the [PGM format](https://users.wpi.edu/~cfurlong/me-593n/pgmimage.html) which also uses 1 byte to represent 8 pixels. This format is useful saving as much space as possible without using a lossy format like JPEG.

An image which resolution of 100 000 x 100 000 pixels multiplied by one bit per pixel gives total number of bits. Dividing by 8 gives total number of bytes. Finally, dividing by 10^9 gives total number of Gigabytes. Multiplied by the expected 1000 images gives the worst case disk usage.

To convert an image into a bit packed from we can use Numpy's packbits feature.

```python
bacteria = packbits(array(Image.open(bacteria_path).convert("1")))
dye = packbits(array(Image.open(dye_path).convert("1")))
```

# Question 2

Images were create with Python's Pillow Image library. Generating the parasite begins with first generating a circle using polar co-ordinates. Given we are calculating the radius for a given angle, it is easy to introduce *noise* in the form of randomly weighted and phase shifted sinusoidal functions. This noise produces the blob shape as opposed to perfect circle. Defining a random offset changes the center of parasite to be more realistic.

```python
def noise(theta, weights, phases):

# https://gamedev.stackexchange.com/questions/62613/need-ideas-for-an-algorithm-to-draw-irregular-blotchy-shapes
    
    return 1 + 0.25*(weights[0]*sin(theta+(2*pi)*phases[0])) \
        + weights[1]*sin(2*theta+2*pi*phases[1]) \
        + weights[2]*sin(3*theta+2*pi*phases[2]) \
        + weights[3]*sin(4*theta+2*pi*phases[3]) \
        + weights[4]*sin(5*theta+2*pi*phases[4])
```

Generating a random number of circles, with random radii in random positions close to center of the parasite (center shifted with random offset) simulates the dye images. 

```python
num_dye_spots = randint(0, 100)
for i in range(num_dye_spots):
    radius = randrange(1, radius_scale//3)
    center_x = randrange(offset, 2*radius_scale + offset)
    center_y = randrange(offset, 2*radius_scale + offset)
    draw.ellipse(xy=(center_x - radius,
                        center_y - radius,
                        center_x + radius,
                        center_y + radius),
                    fill=1)
```

For the sake of rapid prototyping and limitations with the Pillow library, I have kept the resolution of the images to be 10 000 x 10 000. 

![Parasite Example](./examples/parasite-0.jpg)
![Dye Example](./examples/dye-0.jpg)

# Question 3

A single bit represents each pixel. In the parasite image a pixel belonging to the parasite is a 0 (black parasite against white background). Inversely for the dye image where a pixel belonging to the dye is 1. To find the overlap simply invert the parasite pixels and apply the *and* operation to both parasite and dye pixels. By counting the resulting 1 bits we can keep a tally on pixels which overlap. A tally for the number of 1 bits in the parasite image represents the size of the parasite. 

```python
for i in range(len(bacteria)):
    total_bacteria_size += bin(invert(bacteria[i])).count('1')
    total_overlap += bin(bitwise_and(invert(bacteria[i]), dye[i])).count('1')
```

# Question 4

Finding the overlap and counting the 1 bits is an embarrassingly parallel task. Given the operation does not depend on state, there are no issues of concurrency. We can spawn threads to run on a CPU's multiple cores using Python's multiprocessing library. The number of cores on the CPU provides the expected speedup, whilst considering spawning threads comes at the cost of performance overhead. 

Unfortunately, after testing both single and multithreaded performance, I did not see any expected speed up. Instead I saw a performance loss. I believe this is be a language (interpreter) issue as opposed to algorithm. 

```python
def analyse_bacteria(bacteria, dye):
    bacteria_size = bin(invert(bacteria)).count('1') # count number of bacteria pixels inverted
    overlap = bin(bitwise_and(invert(bacteria), dye)).count('1') # count inverted bacteria and dye pixels
    return (bacteria_size, overlap)

def has_cancer_multithreaded(bacteria_path, dye_path):

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
```

