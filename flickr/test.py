import torch
import torch.nn as nn
from scipy.linalg import norm
import skimage.transform
from PIL import Image
import torchvision.transforms as transforms
import numpy
import torch.nn.functional as F
import torchvision.models as models 
import pdb
from .model import *
from .data import *
from .evaluation import *
from .vocab import Vocabulary
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class VGG_feats(nn.Module):
	def __init__(self):
		super(VGG_feats, self).__init__()
		self.cnn = models.__dict__['vgg19'](pretrained=True)
		self.cnn.cuda()
		self.cnn.classifier = nn.Sequential(
	            *list(self.cnn.classifier.children())[:-1])

	def forward(self, image):
		features = self.cnn(image)
		# features = l2norm(features)
		return features

def load_image(file_name):
    """
    Load and preprocess an image
    """
    image = Image.open(file_name)
    # im = numpy.array(image)
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform(image)
im = load_image('/ssd_scratch/cvit/deep/Flickr-8K/Flicker8k_Dataset/106490881_5a2dd9b7bd.jpg')
# root = 'data/f8k'
root = '/ssd_scratch/cvit/deep/Flickr-8K'
# vocab = build_vocab('/ssd_scratch/cvit/deep/Flickr-8K', threshold=4)
with open('vocab/%s_vocab.pkl' %'flickr_precomp', 'rb') as f:
    vocab = pickle.load(f)

# vocab = pickle.load(open(os.path.join(
#     'vocab', '%s_vocab.pkl' % 'f8k_precomp'), 'rb'))
# feats = VGG_feats().cuda()
# feats.eval()
vocab_size = len(vocab)
# print(vocab_size)
# features = feats(im.unsqueeze(0).cuda())
img_enc = EncoderImageFull(1024, use_abs=True).to(device)
txt_enc = EncoderText(vocab_size, 300, 1024, use_abs=False).to(device)
img_enc.eval()
txt_enc.eval()
sub_feat = img_enc(im.unsqueeze(0)).data.cpu().numpy()
train_loader, val_loader = get_loaders(root, vocab, 32, precomp=False)
# savepath = 'model_best.pth.tar'
savepath = 'model_best_full.t7'
checkpoint = torch.load(savepath)

img_enc.load_state_dict(checkpoint['model'][0])
txt_enc.load_state_dict(checkpoint['model'][1])
img_embs, cap_embs = encode_data(txt_enc, img_enc, train_loader)

# sub_feat = img_enc(features).data.cpu().numpy()
with open('data/f8k/f8k_train_caps.txt', 'r') as f:
	lines = f.readlines()
caps = []
for line in lines:
	caption = line.strip().lower()
	caps.append(caption)


scores = numpy.dot(sub_feat, cap_embs.T).flatten()
sorted_args = numpy.argsort(scores)[::-1]
sentences = [caps[a] for a in sorted_args[:10]]
print(sentences)
pdb.set_trace()