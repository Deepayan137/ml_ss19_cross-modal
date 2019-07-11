import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import pickle
import pdb


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        with open(os.path.join(root, '%s.txt'%split)) as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        img_id = index//5
        root = self.root
        image_name = self.lines[index].split(' ')[0] + '.jpg'
        caption = ' '.join(self.lines[index].split(' ')[1:])
        # pdb.set_trace()
        image = Image.open(os.path.join(root, 'Flicker8k_Dataset', image_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.lines)


def collate_fn(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, _  = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, list(ids)

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'f8k_%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'f8k_%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8'))
        caption = []
        # pdb.set_trace()
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length

def get_precomp_loader(data_path, data_split, vocab, batch_size=100,
                       shuffle=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          pin_memory=True,
                                          collate_fn=collate_fn)
    return data_loader



def get_transform(split_name):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(224),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'dev':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

def get_loader_single(root, split, vocab, transform,
                      batch_size=128, shuffle=True,
                      collate_fn=collate_fn):
    
    dataset = FlickrDataset(root=root,
                            split=split,
                            vocab=vocab,
                            transform=transform)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader




def get_loaders(root, vocab, batch_size, precomp=False):
    if precomp == True:
        train_loader = get_precomp_loader(root, 'train', vocab,
                                          batch_size, False)
        val_loader = get_precomp_loader(root, 'dev', vocab,
                                        batch_size, False)
    else:
        transform = get_transform('train')

        train_loader = get_loader_single(root, 'train',
                                        vocab, transform,
                                        batch_size=batch_size, shuffle=True,
                                        collate_fn=collate_fn)

        transform = get_transform('dev')
        
        val_loader = get_loader_single(root, 'dev',
                                    vocab, transform,
                                    batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn)
        transform = get_transform('test')
        
        test_loader = get_loader_single(root, 'test',
                                    vocab, transform,
                                    batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn)


    return train_loader, val_loader


# path = '/ssd_scratch/cvit/deep/Flickr-8K'
# with open('./vocab/%s_vocab.pkl' %'flickr', 'rb') as f:
#     vocab = pickle.load(f)

# train, val = get_loaders(path, vocab, 128)
# for i, batch in enumerate(train):
#     img, targ, lengths = batch
#     pdb.set_trace()
# # data = FlickrDataset(path, 'test', vocab)
