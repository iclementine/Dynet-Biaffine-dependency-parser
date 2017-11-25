import argparse
import numpy as np
from graph import PyGraph
import gensim
from collections import Counter
import sys, random, string

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='+', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=100,
	                    help='Number of dimensions. Default is 100.')

	parser.add_argument('--walk-length', type=int, default=20,
	                    help='Length of walk per source. Default is 20.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=2,
                    	help='Context size for optimization. Default is 2.')

	parser.add_argument('--iter', default=3, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	return parser.parse_args()

ROOT = '<root>'
EOD = '<eod>'
def update_graph(graph, fname):
	nsents = 0
	sent = [[ROOT, ROOT, 0, ROOT]]
	for line in open(fname).readlines():
		info = line.strip().split()
		if info:
			assert(len(info)==10), 'Illegal line: %s'%line
			word, tag, head, rel = info[1].lower(), info[3], int(info[6]), info[7]
			sent.append([word, tag, head, rel])
		else:
			for idx, (word, tag, head, rel) in enumerate(sent[1:],1):
				depth = 1
				h = head
				while h!=0:
					h = sent[h][2]
					depth +=1
			nsents += 1
			head_set = set([ head for word, _, head, _ in sent[1:]])
			for head in xrange(len(sent)):
				if head not in head_set:
					graph[(sent[head][0],EOD)] += 1
			#for word, _, head, _ in sent[1:]:
			#	if word in string.punctuation:
			#		print sent[head][0]
			graph.update([(sent[head][0], word) for word, _, head, _ in sent[1:]])
			sent = [[ROOT, ROOT, 0, ROOT]]
	return nsents

def read_graph(file_list):
	graph = Counter()
	nsents = 0
	for fname in file_list:
		nsents += update_graph(graph, fname)
	print 'number of sentences', nsents
	u, v, w = [], [], []
	vocab = set()
	for edge in graph:
		if edge[0] in string.punctuation or edge[1] in string.punctuation:
			continue
		if graph[edge] >=3:
			u.append(edge[0])
			v.append(edge[1])
			w.append(graph[edge])
			vocab.add(edge[0])
			vocab.add(edge[1])
	id2word = list(vocab)
	word2id = dict(zip(id2word,range(len(id2word))))
	u = [word2id[x] for x in u]
	v = [word2id[x] for x in v]
	G = PyGraph(u, v, w)
	return G, id2word, word2id, nsents

class Simulate_walks(object):
	def __init__(self, G, id2word, word2id, num_walks, walk_length):
		self.G = G
		self.id2word = id2word
		self.word2id = word2id
		self.num_walks = num_walks
		self.walk_length = walk_length
	
	def __iter__(self):
		node = self.word2id[ROOT]
		for walk_iter in range(self.num_walks):
			out = [ self.id2word[x] for x in self.G.walk(start_node=node, walk_length=self.walk_length)]
			#print out
			yield out

def main(args):
	G, id2word, word2id, nsents = read_graph(args.input)
	G.preprocess()
	print 'Graph preprocessed'
	walks = Simulate_walks(G, id2word, word2id, args.num_walks * nsents, args.walk_length)
	model = gensim.models.Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)

if __name__ == "__main__":
	args = parse_args()
	main(args)
