import requests as req
import argparse
import os 

def get_dataset(name):
    
    if not os.path.isdir(name):
        os.mkdir(name)

    print(" Selected Dataset : ", name)
    if name == 'stanford40':
        urls = []
        urls.append('http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip')
        urls.append('http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip')
    else:
        NotImplementedError

    for url in urls:
        file = req.get(url)
        out_path = os.path.join(
            name, url.split('/')[-1]
        )
        with open(out_path, 'wb') as f:
            f.write(file.content)


if __name__ == "__main__":
    # arg parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stanford40', choices=['stanford40','voc','willow','ppmi'])
    opt = parser.parse_args()
    get_dataset(opt.dataset)
