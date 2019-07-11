import os
import pickle
import torch

# from vocab import Vocabulary
from .model import *
from .data import *
from .evaluation import *
from tqdm import *
from .vocab import build_vocab
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")

import pdb

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = 0.0002 * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
    	param_group['lr'] = lr

def validate(val_loader, txt_enc, img_enc):
	img_embs, cap_embs = encode_data(txt_enc, img_enc, val_loader)
	r1, r5, r10, medr = i2t(img_embs, cap_embs)
	r1i, r5i, r10i, medri = t2i(img_embs, cap_embs)
	score = r1 + r5 + r10 + r1i + r5i + r10i
	return r1

def main():
	root = '/ssd_scratch/cvit/deep/Flickr-8K'
	# root = 'data/f8k'
	# with open('vocab/%s_precomp_vocab.pkl' %'f8k', 'rb') as f:
	# 	vocab = pickle.load(f)
	
	if not os.path.exists('vocab/%s_precomp_vocab.pkl' %'flickr'):
		vocab = build_vocab('/ssd_scratch/cvit/deep/Flickr-8K', threshold=4)
		with open('vocab/%s_precomp_vocab.pkl' %'flickr', 'wb') as f:
			pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
	else:
		with open('vocab/%s_precomp_vocab.pkl' %'flickr', 'rb') as f:
			vocab = pickle.load(f)
	batch_size = 32
	vocab_size = len(vocab)
	print('Dictionary size: ' + str(vocab_size))
	embed_size = 1024
	img_dim = 4096
	word_dim = 300
	num_epochs = 100
	train_loader, val_loader = get_loaders(root, vocab, batch_size, precomp=False)
	img_enc = EncoderImageFull(embed_size, use_abs=True).to(device)
	# img_enc = EncoderImagePrecomp(img_dim, embed_size, use_abs=False).to(device)
	txt_enc = EncoderText(vocab_size, word_dim, embed_size, use_abs=False).to(device)
	savepath = 'model_best_full.t7'
	start_epoch, best_rsum = 0, 0

	if os.path.isfile(savepath):
		checkpoint = torch.load(savepath)
		start_epoch = checkpoint['epoch']
		best_rsum = checkpoint['best_rsum']
		print('Loaded checkpoint at {} trained for {} epochs'.format(savepath, checkpoint['epoch']))
		img_enc.load_state_dict(checkpoint['model'][0])
		txt_enc.load_state_dict(checkpoint['model'][1])

	# criterion = ContrastiveLoss(margin=0.2, measure='cosine', max_violation=True)
	params = list(txt_enc.parameters())
	params += list(img_enc.fc.parameters())
	criterion = PairwiseRankingLoss(margin=0.2)
	optimizer = torch.optim.Adam(params, lr=0.0002)
	rsum = validate(val_loader, txt_enc, img_enc)
	for epoch in range(start_epoch, num_epochs):
		adjust_learning_rate(optimizer, epoch)
		for i, batch in enumerate(tqdm(train_loader)):
			images, captions, lengths, ids = batch
			images = images.to(device)
			captions = captions.to(device)
			img_emb = img_enc(images)
			cap_emb = txt_enc(captions, lengths)
			loss = criterion(img_emb, cap_emb)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
				
		rsum = validate(val_loader, txt_enc, img_enc)
		print('Epochs: [%d]/[%d] AvgScore: %.2f Loss: %.2f'%(epoch, num_epochs, rsum, loss.item()))
		is_best = rsum > best_rsum
		best_rsum = max(rsum, best_rsum)
		save_checkpoint({
						'epoch': epoch + 1,
						'model': [img_enc.state_dict(), txt_enc.state_dict()],
						'best_rsum': best_rsum
						}, savepath, is_best) 

if __name__ == '__main__':
	main()