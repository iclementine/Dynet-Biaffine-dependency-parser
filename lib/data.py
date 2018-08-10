# -*- coding: UTF-8 -*-
from __future__ import division
from collections import Counter
import numpy as np

from lib.k_means import KMeans

class Vocab(object):
	PAD, ROOT, UNK = 0, 1, 2

	def __init__(self, input_file, pret_file = None, min_occur_count = 2):
		word_counter = Counter()
		tag_set = set()
		rel_set = set()
		with open(input_file) as f:
			for line in f.readlines():
				info = line.strip().split()
				if info:
					# 这么说可以肯定 input_file 应该是 CoNLL-U 格式的了
					assert(len(info)==10), 'Illegal line: %s'%line 
					word, tag, head, rel = info[1].lower(), info[3], int(info[6]), info[7]
					word_counter[word] += 1
					tag_set.add(tag)
					if rel !='root':
						rel_set.add(rel) # 'root' 就是一个锚点， 并不像其他的 rel 那么有意义

		self._id2word = ['<pad>', '<root>', '<unk>']
		self._id2tag = ['<pad>', '<root>', '<unk>']
		self._id2rel = ['<pad>', 'root'] # 吼呀， 果然有这样的实现， 给我的感觉就是不要怂， 直接造轮子
		for word, count in word_counter.most_common():
			if count > min_occur_count:
				self._id2word.append(word)
		
		self._pret_file = pret_file
		if pret_file:
			self._add_pret_words(pret_file)
		self._id2tag += list(tag_set)
		self._id2rel += list(rel_set)

		reverse = lambda x : dict(zip(x, range(len(x))))
		self._word2id = reverse(self._id2word)
		self._tag2id = reverse(self._id2tag)
		self._rel2id = reverse(self._id2rel)
		
		self._words_in_train_data = len(self._id2word) # moved from add_pret_words
		print("Vocab info: #words %d, #tags %d #rels %d"%(self.vocab_size,self.tag_size, self.rel_size))
		
	def _add_pret_words(self, pret_file):
		# pret_file 是一个 path
		# moved to init
		print('#words in training set:', self._words_in_train_data)
		words_in_train_data = set(self._id2word)
		with open(pret_file) as f:
			for line in f.readlines():
				line = line.strip().split()
				if line:
					word = line[0]
					if word not in words_in_train_data:
						self._id2word.append(word)
						# 这样子不是可能会扩大了词典吗？， 不过没有关系 reverse 发生在吼哦免
		#print 'Total words:', len(self._id2word)

	def get_pret_embs(self):
		assert (self._pret_file is not None), "No pretrained file provided."
		embs = [[]] * len(self._id2word) # 这个的词数可能比 words_in_train 大一点
		with open(self._pret_file) as f:
			for line in f.readlines():
				line = line.strip().split()
				if line:
					word, data = line[0], line[1:]
					embs[self._word2id[word]] = data # 这里不会出现 key error 因为词典扩张过了
		emb_size = len(data)
		# 如果train 中的某个词在 pretrain 中没有出现， 用 0 代替
		for idx, emb in enumerate(embs):
			if not emb:
				embs[idx] = np.zeros(emb_size)
		pret_embs = np.array(embs, dtype=np.float32)
		return pret_embs / np.std(pret_embs)


	def get_word_embs(self, word_dims):
		# 如果有 pretrain, 就随机初始化自己的 word_embeds
		# 如果没有直接用 0 的哦， 有点凶残
		if self._pret_file is not None:
			return np.random.randn(self.words_in_train, word_dims).astype(np.float32)
		return np.zeros((self.words_in_train, word_dims), dtype=np.float32)

	def get_tag_embs(self, tag_dims):
		return np.random.randn(self.tag_size, tag_dims).astype(np.float32)

	def word2id(self, xs):
		# 相当于重载的函数了， 既可以 str -> float, 又可以 list(str) -> list(int)
		if isinstance(xs, list):
			return [self._word2id.get(x, self.UNK) for x in xs]
		return self._word2id.get(xs, self.UNK)
	
	def id2word(self, xs):
		if isinstance(xs, list):
			return [self._id2word[x] for x in xs]
		return self._id2word[xs]

	def rel2id(self, xs):
		if isinstance(xs, list):
			return [self._rel2id[x] for x in xs]
		return self._rel2id[xs]

	def id2rel(self,xs):
		if isinstance(xs, list):
			return [self._id2rel[x] for x in xs]
		return self._id2rel[xs]

	def tag2id(self, xs):
		if isinstance(xs, list):
			return [self._tag2id.get(x, self.UNK) for x in xs]
		return self._tag2id.get(xs, self.UNK)

	@property 
	def words_in_train(self):
		#　这个不会扩张
		return self._words_in_train_data

	@property
	def vocab_size(self):
		# 这个可能扩张过
		return len(self._id2word)

	@property
	def tag_size(self):
		return len(self._id2tag)

	@property
	def rel_size(self):
		return len(self._id2rel)

class DataLoader(object):
	def __init__(self, input_file, n_bkts, vocab):
		sents = []
		sent = [[Vocab.ROOT, Vocab.ROOT, 0, Vocab.ROOT]] # 这是 ROOT 节点的表示
		with open(input_file) as f:
			for line in f.readlines():
				info = line.strip().split()
				if info:
					assert(len(info)==10), 'Illegal line: %s'% line
					# 虽然有很多行， 但是能用于自动句法解析的基本也就只有这几列， word 事实上已经是 lemma 了
					word, tag, head, rel = vocab.word2id(info[1].lower()), vocab.tag2id(info[3]), int(info[6]), vocab.rel2id(info[7])
					sent.append([word, tag, head, rel])
				else:
					sents.append(sent)
					sent = [[Vocab.ROOT, Vocab.ROOT, 0, Vocab.ROOT]]
		
		len_counter = Counter()
		for sent in sents:
			len_counter[len(sent)] += 1	
		self._bucket_sizes = KMeans(n_bkts, len_counter).splits # 类型是什么 应该是 int
		self._buckets = [[] for i in range(n_bkts)]
		len2bkt = {}
		prev_size = -1
		for bkt_idx, size in enumerate(self._bucket_sizes):
			len2bkt.update(zip(range(prev_size+1, size+1), [bkt_idx] * (size - prev_size)))
			prev_size = size

		self._record = []
		for sent in sents:
			bkt_idx = len2bkt[len(sent)]
			self._buckets[bkt_idx].append(sent)
			idx = len(self._buckets[bkt_idx]) - 1
			self._record.append((bkt_idx, idx))

		for bkt_idx, (bucket, size) in enumerate(zip(self._buckets, self._bucket_sizes)):
			self._buckets[bkt_idx] = np.zeros((size, len(bucket), 4), dtype=np.int32)
			for idx, sent in enumerate(bucket):
				self._buckets[bkt_idx][:len(sent), idx, :] = np.array(sent, dtype=np.int32)
				
	@property
	def idx_sequence(self):
		return [x[1] for x in sorted(zip(self._record, range(len(self._record))))]
	
	def get_batches(self, batch_size, shuffle= True):
		batches = []
		for bkt_idx, bucket in enumerate(self._buckets):
			bucket_len = bucket.shape[1]
			n_tokens = bucket_len * self._bucket_sizes[bkt_idx]
			n_splits = max(n_tokens // batch_size, 1)
			range_func = np.random.permutation if shuffle else np.arange
			for bkt_batch in np.array_split(range_func(bucket_len), n_splits):
				batches.append((bkt_idx, bkt_batch))
		if shuffle:
			np.random.shuffle(batches)

		for bkt_idx, bkt_batch in batches:
			word_inputs = self._buckets[bkt_idx][:,bkt_batch, 0] # fancy indexing
			tag_inputs = self._buckets[bkt_idx][:,bkt_batch, 1]
			arc_targets = self._buckets[bkt_idx][:,bkt_batch, 2]
			rel_targets = self._buckets[bkt_idx][:,bkt_batch, 3]
			yield word_inputs, tag_inputs, arc_targets, rel_targets
			# 每一个的形状都是 (bucket_sizes[bkt_idx], len(bkt_batch))
