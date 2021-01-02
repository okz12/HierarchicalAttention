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
        encoded = self.word_tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences[:max_sentences],
                                                        add_special_tokens=False,
                                                        padding='max_length', truncation=True, max_length=max_words,
                                                        return_tensors='pt')
        tokens, att = encoded['input_ids'], encoded['attention_mask']
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
        train_dataset = None,
        valid_dataset = None,
        test_dataset = None
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

        # Sentence-level attention module (which will, in-turn, contain the word-level attention module)
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


        # Classifier
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
        # Apply sentence-level attention module (and in turn, word-level attention module) to get document embeddings
        document_embeddings, word_alphas, sentence_alphas = self.sentence_attention(
            documents, sentences_per_document, words_per_sentence
        )
        # (n_documents, 2 * sentence_rnn_size), (n_documents, max(sentences_per_document), max(words_per_sentence)), (n_documents, max(sentences_per_document))

        # Classify
        scores = self.fc(self.dropout(document_embeddings))  # (n_documents, n_classes)

        return scores, word_alphas, sentence_alphas

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
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
        self.log('train_loss', loss, prog_bar=False)
        self.log('train_acc', acc, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        acc, loss = self.step(batch)
        self.log('test_loss', loss, prog_bar=False)
        self.log('test_acc', acc, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        acc, loss = self.step(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=32, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, shuffle=False)

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

        # Word-level attention module
        self.word_attention = WordAttention(
            embedding_layer,
            embedding_size,
            fine_tune_embeddings,
            word_rnn_size,
            word_rnn_layers,
            word_att_size,
            dropout,
        )

        # Bidirectional sentence-level RNN
        self.sentence_rnn = nn.GRU(
            input_size=2 * word_rnn_size,
            hidden_size=sentence_rnn_size,
            num_layers=sentence_rnn_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        # Sentence-level attention network
        self.sentence_attention = nn.Linear(
            in_features=2 * sentence_rnn_size, out_features=sentence_att_size
        )

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(
            in_features=sentence_att_size, out_features=1, bias=False
        )  # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """

        # Re-arrange as sentences by removing sentence-pads (DOCUMENTS -> SENTENCES)
        packed_sentences = pack_padded_sequence(
            input=documents,
            lengths=sentences_per_document.tolist(),
            batch_first=True,
            enforce_sorted=False,
        )  # a PackedSequence object, where 'data' is the flattened sentences (n_sentences, word_pad_len)

        # Re-arrange sentence lengths in the same way (DOCUMENTS -> SENTENCES)
        packed_words_per_sentence = pack_padded_sequence(
            input=words_per_sentence,
            lengths=sentences_per_document.tolist(),
            batch_first=True,
            enforce_sorted=False,
        )  # a PackedSequence object, where 'data' is the flattened sentence lengths (n_sentences)

        # Find sentence embeddings by applying the word-level attention module
        sentences, word_alphas = self.word_attention(
            packed_sentences.data, packed_words_per_sentence.data
        )  # (n_sentences, 2 * word_rnn_size), (n_sentences, max(words_per_sentence))
        sentences = self.dropout(sentences)

        # Apply the sentence-level RNN over the sentence embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_sentences, _ = self.sentence_rnn(
            PackedSequence(
                data=sentences,
                batch_sizes=packed_sentences.batch_sizes,
                sorted_indices=packed_sentences.sorted_indices,
                unsorted_indices=packed_sentences.unsorted_indices,
            )
        )  # a PackedSequence object, where 'data' is the output of the RNN (n_sentences, 2 * sentence_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_s = self.sentence_attention(
            packed_sentences.data
        )  # (n_sentences, att_size)
        att_s = torch.tanh(att_s)  # (n_sentences, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s).squeeze(1)  # (n_sentences)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over sentences in the same document

        # First, take the exponent
        max_value = (
            att_s.max()
        )  # scalar, for numerical stability during exponent calculation
        att_s = torch.exp(att_s - max_value)  # (n_sentences)

        # Re-arrange as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        att_s, _ = pad_packed_sequence(
            PackedSequence(
                data=att_s,
                batch_sizes=packed_sentences.batch_sizes,
                sorted_indices=packed_sentences.sorted_indices,
                unsorted_indices=packed_sentences.unsorted_indices,
            ),
            batch_first=True,
        )  # (n_documents, max(sentences_per_document))

        # Calculate softmax values as now sentences are arranged in their respective documents
        sentence_alphas = att_s / torch.sum(
            att_s, dim=1, keepdim=True
        )  # (n_documents, max(sentences_per_document))

        # Similarly re-arrange sentence-level RNN outputs as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        documents, _ = pad_packed_sequence(
            packed_sentences, batch_first=True
        )  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)

        # Find document embeddings
        documents = documents * sentence_alphas.unsqueeze(
            2
        )  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)
        documents = documents.sum(dim=1)  # (n_documents, 2 * sentence_rnn_size)

        # Also re-arrange word_alphas (SENTENCES -> DOCUMENTS)
        word_alphas, _ = pad_packed_sequence(
            PackedSequence(
                data=word_alphas,
                batch_sizes=packed_sentences.batch_sizes,
                sorted_indices=packed_sentences.sorted_indices,
                unsorted_indices=packed_sentences.unsorted_indices,
            ),
            batch_first=True,
        )  # (n_documents, max(sentences_per_document), max(words_per_sentence))

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

        # Fine tune embedding layer
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune_embeddings

        # Bidirectional word-level RNN
        self.word_rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=word_rnn_size,
            num_layers=word_rnn_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        # Word-level attention network
        self.word_attention = nn.Linear(
            in_features=2 * word_rnn_size, out_features=word_att_size
        )

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(
            in_features=word_att_size, out_features=1, bias=False
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, words_per_sentence):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """

        # Get word embeddings, apply dropout
        sentences = self.embeddings(sentences)
        if type(sentences) is tuple:
            sentences = sentences[0]
        sentences = self.dropout(sentences)  # (n_sentences, word_pad_len, emb_size)

        # Re-arrange as words by removing word-pads (SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(
            sentences,
            lengths=words_per_sentence.tolist(),
            batch_first=True,
            enforce_sorted=False,
        )  # a PackedSequence object, where 'data' is the flattened words (n_words, word_emb)

        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_words, _ = self.word_rnn(
            packed_words
        )  # a PackedSequence object, where 'data' is the output of the RNN (n_words, 2 * word_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(packed_words.data)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = (att_w.max())  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(
            PackedSequence(
                data=att_w,
                batch_sizes=packed_words.batch_sizes,
                sorted_indices=packed_words.sorted_indices,
                unsorted_indices=packed_words.unsorted_indices,
            ),
            batch_first=True,)  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(packed_words, batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        return sentences, word_alphas