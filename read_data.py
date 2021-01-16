import os, sys, glob
from PIL import Image, ImageFile, ImageOps
import numpy as np
from tqdm import tqdm

# Number of instances
num_instance={"black":0,
              "mg":0,
              "moth":0,
              "oil":0,
              "healthy":0,
              "background":0}
dict_name={"b":"black","g":"mg","m":"moth","o":"oil","h":"healthy"}

def mlEncoder(fname):
    """
        b: black
        g: mg
        m: moth
        o: oil
        h: healthy
        File name format: bgo_123.jpg
    """
    label_list = [0,0,0,0,0]
    dict_label = {"b":0, "g":1, "m":2, "o":3}
    for l in fname.split("_")[0]:
        #print(l)
        # Calculate instances
        num_instance[dict_name[l]]+=1
        label_list[dict_label[l]]=1
        
    #print(f"File:{fname}, label:{label_list}")
    return label_list

def slEncoder(folder):
    """
        Label name will be same as folder name
    """
    if folder=="healthy":
        num_instance["healthy"]+=1
        label_list = [0,0,0,0,0]
        return label_list   
    label_list = [0,0,0,0,0]
    dict_label = {"black":0, "mg":1, "moth":2, "oil":3, "background":4}
    # Calculate instances
    num_instance[folder]+=1
    label_list[dict_label[folder]]=1
    return label_list

def load_image_file(file, mode='RGB', size=None):
    # Load the image with PIL
    img = Image.open(file)
    img = img.convert(mode)
    if size:
        if type(size) is not tuple:
            print("Wrong type of size")
        else:
            img = img.resize(size)
    return img

def read_dataset(loc, aug=False, input_shape=(224, 224)):
    image, label = [], []
    for folders in glob.glob(loc):
        print(f"Loading {folders}:")
        for f in tqdm(os.listdir(folders)):
            print(f"\rLoading data: {os.path.join(folders,f)}", end="")
            if os.path.basename(folders)=="multi":
                # ml
                img_label = mlEncoder(f)
                label.append(img_label)
                img = load_image_file(os.path.join(folders,f), size = (224,224))
                image.append(np.array(img))
            else:
                # sl
                img_label = slEncoder(os.path.basename(folders))
                label.append(img_label)
                img = load_image_file(os.path.join(folders,f), size = (224,224))
                image.append(np.array(img))

    # image and label is x and y
    x = np.array(image, dtype=np.float16) / 255.0
    y = np.array(label, dtype=np.float16)
    print("Instances: ")
    print(num_instance)
    return x, y