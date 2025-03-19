import os 
import torch 
import pandas as pd 
import nltk # Natural Language Toolkit
from nltk import word_tokenize
nltk.download('punkt') # Download the punkt tokenizer
from torch.utils.data import Dataset, DataLoader 


class Vocabulary:
    def __init__(self, freq_threshold):

        # We need a way to convert from word to index and vice versa 
        self.index_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.freq_threshold = freq_threshold 

    def __len__(self):
        return len(self.index_to_word)

    @staticmethod
    def tokenizer(text):
        return [wordfor word in word_tokenize(text.lower())]

    def build_vocab(self, sentence_list):

        frequencies = {}
        start_idx = 4 # Start index of the vocabulary

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            for token in tokens:
                if token not in frequencies:
                    frequecies[token] = 1
                else:
                    frequencies[token] += 1
                    if freq_threshold[token] == selffreq_threshold:
                        self.word_to_index[token] = start_idx
                        self.index_to_word[start_idx] = token
                        start_idx += 1
                    else:
                        continue

    def get_indices(sentence):
        '''
        Turn a sentence into a list of indices.

        Each word corresponds to an indices in the word_to_index dictionary we have built above 
        If a word does not exist then we return it as an Unkown ("UNK") token
        '''
        tokenized_text = self.tokenizer(sentence_list)
        return = [
            self.word_to_index[word] if word in self.word_to_index else self.word_to_index["UNK"]
            for word in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, caption_file, transforms, freq_threshold=5):
        self.root_dir = root_dir 
        self.df = caption_file
        self.transforms = transforms # Image transformations

        self.img_path = sorted(self.df["image"])
        self.captions = sorted(self.df["caption"])

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __init__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.img_path[idx]
        caption = self.img(idx)

        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if transforms:
            img = self.transforms(img)            

        numericalized_caption = self.vocab.word_to_index["<SOS>"]
        numericalized_caption += self.vocab.get_indices(caption)
        numericalized_caption.append(self.vocab.word_to_index["<EOS>"])

        # Image will be a torch tensor because it will be included in our transforms, we need to convert caption
        return img, torch.tensor(numericalized_caption)






        return img, caption


