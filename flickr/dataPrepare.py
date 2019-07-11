import os
import cv2
import string
from collections import defaultdict
import pdb
	

def clean_descriptions(desc):
	desc = desc.lower()
	desc = desc.translate(str.maketrans('', '', string.punctuation))
	desc = desc.split()
	desc = [word for word in desc if word.isalpha()]
	desc = ' '.join(desc)
	return desc

def load_descriptions(doc):
	mapping = defaultdict(list)
	with open(doc, 'r') as f:
		text = f.read()
	for line in text.split('\n'):
		tokens = line.split()
		if len(line) < 2:
			continue
		image_id, image_desc = tokens[0], tokens[1:]
		image_id = image_id.split('.')[0]
		
		image_desc = ' '.join(image_desc)
		mapping[image_id].append(clean_descriptions(image_desc))
	return mapping

def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
# parse descriptions
from tqdm import *
splits = ['train', 'dev', 'test']
doc = 'Flickr8k.token.txt'
path = '/ssd_scratch/cvit/deep/Flickr-8K/'
doc_path = os.path.join(path, doc)
descriptions = load_descriptions(doc_path)
for split in splits:
	filename = 'Flickr_8k.%sImages.txt'%split
	with open(os.path.join(path, filename), 'r') as f:
		lines = f.readlines()
	for line in tqdm(lines):
		id_ = line.split('.')[0]
		descs = descriptions[id_]
		for desc in descs:
			# pdb.set_trace()
			with open(path+'/'+split+'.txt', 'a') as f_out:
				f_out.write('{} {}\n'.format(id_, desc))


save_descriptions(descriptions, '%s.txt'%split)


