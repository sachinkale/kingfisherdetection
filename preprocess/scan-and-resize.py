import os
import re
from PIL import Image, ImageOps
desired_size = 224

dirlist = ["/Users/xxx/work/scraping/train3_cropped","/Users/xxx/work/scraping/test3_cropped"]


def resize(im_pth,n_pth):
    im = Image.open(im_pth)
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                      (desired_size-new_size[1])//2))

    #new_im.show()
    new_im.save(n_pth)

for dir in dirlist:
    save_dir = dir + "_resized"
    types = os.listdir(dir)

    for t in types:

        path = dir + "/" + t
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                filepath = os.path.join(path, file)
                os.makedirs(save_dir + "/" + t, exist_ok=True)  # succeeds even if directory exists.

                new_filepath = save_dir + "/" + t + "/" + file
                if (re.search('jpg', filepath)):
                
                    resize(filepath,new_filepath)
