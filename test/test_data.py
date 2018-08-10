import os

from lib import Vocab, DataLoader

root_path = '/home/clementine/projects/树库转换/ctb51_processed/dep_zhang_clark_conlu/'
training = os.path.join(root_path, 'train.txt.3.pa.gs.tab.conllu')
vocab = Vocab(training, pret_file=None, min_occur_count=2)

ctb_loader = DataLoader(training, 40, vocab)
for i, example in enumerate(ctb_loader.get_batches(40, shuffle=True)):
	print(example[0].shape, example[1].shape, example[2].shape, example[3].shape)
	if i == 0:
		break
