import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from typing import List


class Preprocessor:
    def __init__(self, sentence_tokenizer, word_tokenizer):
        self.sentence_tokenizer = sentence_tokenizer
        self.word_tokenizer = word_tokenizer

    def encode_document(self, document: List[List[str]], max_words: int = 20, max_sentences: int = 15):
        sentences = self.sentence_tokenizer.tokenize(document)
        encoded = self.word_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=sentences[:max_sentences],
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_words,
            return_tensors="pt",
        )
        tokens, att = encoded["input_ids"], encoded["attention_mask"]
        sentence_pad_length = max_sentences - tokens.shape[0]
        tokens = nn.ZeroPad2d((0, 0, 0, sentence_pad_length))(tokens)
        sentences = min(max_sentences, len(sentences))
        words_per_sentence = att.sum(dim=1)
        words_per_sentence = nn.ConstantPad1d((0, sentence_pad_length), 0)(words_per_sentence)
        return tokens, sentences, words_per_sentence


class HierarchicalAttentionNetwork(pl.LightningModule):
    """
    The overarching Hierarchial Attention Network (HAN).
    """

    def __init__(
        self,
        n_classes,
        embedding_layer,
        embedding_size,
        fine_tune_embeddings,
        word_rnn_size,
        sentence_rnn_size,
        word_rnn_layers,
        sentence_rnn_layers,
        word_att_size,
        sentence_att_size,
        dropout=0.5,
        train_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        batch_size=32,
        lr=1e-3,
    ):
        """
        :param n_classes: number of classes
        :param vocab_size: number of words in the vocabulary of the model
        :param embedding_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super(HierarchicalAttentionNetwork, self).__init__()
        # PL elements
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr

        self.sentence_attention = SentenceAttention(
            embedding_layer,
            embedding_size,
            fine_tune_embeddings,
            word_rnn_size,
            sentence_rnn_size,
            word_rnn_layers,
            sentence_rnn_layers,
            word_att_size,
            sentence_att_size,
            dropout,
        )

        self.fc = nn.Linear(2 * sentence_rnn_size, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        document_embeddings, word_alphas, sentence_alphas = self.sentence_attention(
            documents, sentences_per_document, words_per_sentence
        )

        scores = self.fc(self.dropout(document_embeddings))

        return scores, word_alphas, sentence_alphas

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer

    def step(self, batch):
        documents, sentences_per_document, words_per_sentence, labels = batch
        sentences_per_document = sentences_per_document.squeeze(1)
        words_per_sentence = words_per_sentence
        labels = labels.squeeze(1)
        scores, word_alphas, sentence_alphas = self(documents, sentences_per_document, words_per_sentence)
        loss = F.cross_entropy(scores, labels)
        preds = torch.argmax(scores, dim=1)
        acc = accuracy(preds, labels)
        return acc, loss

    def training_step(self, batch, batch_idx):
        acc, loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", acc, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        acc, loss = self.step(batch)
        self.log("test_loss", loss, prog_bar=False)
        self.log("test_acc", acc, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        acc, loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class SentenceAttention(nn.Module):
    """
    The sentence-level attention module.
    """

    def __init__(
        self,
        embedding_layer,
        embedding_size,
        fine_tune_embeddings,
        word_rnn_size,
        sentence_rnn_size,
        word_rnn_layers,
        sentence_rnn_layers,
        word_att_size,
        sentence_att_size,
        dropout,
    ):
        """
        :param embedding_layer: word embedding layer from pre-trained model
        :param embedding_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super(SentenceAttention, self).__init__()

        self.word_attention = WordAttention(
            embedding_layer,
            embedding_size,
            fine_tune_embeddings,
            word_rnn_size,
            word_rnn_layers,
            word_att_size,
            dropout,
        )
        self.sentence_rnn = nn.GRU(
            input_size=2 * word_rnn_size,
            hidden_size=sentence_rnn_size,
            num_layers=sentence_rnn_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.sentence_attention = nn.Linear(in_features=2 * sentence_rnn_size, out_features=sentence_att_size)
        self.sentence_context_vector = nn.Linear(in_features=sentence_att_size, out_features=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """

        # on packing / padding sequences: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

        # Apply word attention to get sentence embeddings
        packed_sentences = pack_padded_sequence(
            input=documents,
            lengths=sentences_per_document.tolist(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_words_per_sentence = pack_padded_sequence(
            input=words_per_sentence,
            lengths=sentences_per_document.tolist(),
            batch_first=True,
            enforce_sorted=False,
        )
        sentences, word_alphas = self.word_attention(packed_sentences.data, packed_words_per_sentence.data)
        sentences = self.dropout(sentences)

        # Pack sentences to apply attention
        packed_sentences = PackedSequence(
                data=sentences,
                batch_sizes=packed_sentences.batch_sizes,
                sorted_indices=packed_sentences.sorted_indices,
                unsorted_indices=packed_sentences.unsorted_indices,
            )
        packed_sentences, _ = self.sentence_rnn(packed_sentences)

        # Apply sentence attention
        att_s = torch.tanh(self.sentence_attention(packed_sentences.data))
        att_s = self.sentence_context_vector(att_s).squeeze(1)
        att_s = torch.exp(att_s - att_s.max())

        packed_att = PackedSequence(
                data=att_s,
                batch_sizes=packed_sentences.batch_sizes,
                sorted_indices=packed_sentences.sorted_indices,
                unsorted_indices=packed_sentences.unsorted_indices,
            )
        att_s, _ = pad_packed_sequence(packed_att, batch_first=True)

        # Generate document embedding
        sentence_alphas = att_s / torch.sum(att_s, dim=1, keepdim=True)
        documents, _ = pad_packed_sequence(packed_sentences, batch_first=True)
        documents = documents * sentence_alphas.unsqueeze(2)
        documents = documents.sum(dim=1)

        packed_word_alphas = PackedSequence(
            data=word_alphas,
            batch_sizes=packed_sentences.batch_sizes,
            sorted_indices=packed_sentences.sorted_indices,
            unsorted_indices=packed_sentences.unsorted_indices,
        )
        word_alphas, _ = pad_packed_sequence(packed_word_alphas, batch_first=True)

        return documents, word_alphas, sentence_alphas


class WordAttention(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(
        self,
        embedding_layer,
        embedding_size,
        fine_tune_embeddings,
        word_rnn_size,
        word_rnn_layers,
        word_att_size,
        dropout,
    ):
        """
        :param embedding_layer: word embedding layer from pre-trained model
        :param embedding_size: size of word embeddings
        :param fine_tune_embeddings: retrain embedding layer
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param word_att_size: size of word-level attention layer
        :param dropout: dropout
        """
        super(WordAttention, self).__init__()

        self.embeddings = embedding_layer
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune_embeddings
        self.word_rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=word_rnn_size,
            num_layers=word_rnn_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.word_attention = nn.Linear(in_features=2 * word_rnn_size, out_features=word_att_size)
        self.word_context_vector = nn.Linear(in_features=word_att_size, out_features=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, words, words_per_sentence):
        """
        Forward propagation.

        :param words: tokenized and encoded sentences, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """

        # Get word embeddings
        words = self.embeddings(words)
        if type(words) == tuple:
            words = words[0]
        words = self.dropout(words)

        # Pack words to apply attention
        packed_words = pack_padded_sequence(
            words,
            lengths=words_per_sentence.tolist(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_words, _ = self.word_rnn(packed_words)

        # Apply word attention
        att_w = torch.tanh(self.word_attention(packed_words.data))
        att_w = self.word_context_vector(att_w).squeeze(1)
        att_w = torch.exp(att_w - att_w.max())

        att_w, _ = pad_packed_sequence(
            PackedSequence(
                data=att_w,
                batch_sizes=packed_words.batch_sizes,
                sorted_indices=packed_words.sorted_indices,
                unsorted_indices=packed_words.unsorted_indices,
            ),
            batch_first=True,
        )

        # Generate sentence embeddings
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)
        sentences, _ = pad_packed_sequence(packed_words, batch_first=True)
        sentences = sentences * word_alphas.unsqueeze(2)
        sentences = sentences.sum(dim=1)

        return sentences, word_alphas
