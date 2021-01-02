import torch
from torch.utils.data import Dataset
import time

class HANDataset(Dataset):

    def __init__(self, features, labels):
        self.features, self.sentences_per_document, self.words_per_sentence = zip(*features)
        self.labels = labels

    def __getitem__(self, i):
        return (torch.LongTensor(self.features[i]),
                torch.LongTensor([self.sentences_per_document[i]]), 
                torch.LongTensor(self.words_per_sentence[i]),
                torch.LongTensor([self.labels[i]]))

    def __len__(self):
        return len(self.labels)

def save_checkpoint(epoch, model, optimizer, pt_model, rev_label_map, filename = 'checkpoint_han.pth.tar'):
    state = {'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'pt_model': pt_model,
            'rev_label_map': rev_label_map}
    torch.save(state, filename)

