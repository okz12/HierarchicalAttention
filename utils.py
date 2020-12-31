import torch
from torch.utils.data import Dataset
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    start = time.time()

    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_loader):


        documents = documents.to(device)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)
        words_per_sentence = words_per_sentence.to(device)
        labels = labels.squeeze(1).to(device)

        scores, word_alphas, sentence_alphas = model(documents, sentences_per_document, words_per_sentence)
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()

        # grad clip?

        optimizer.step()

        _, predictions = scores.max(dim=1)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        losses.update(loss.item(), labels.size(0))
        accs.update(accuracy, labels.size(0))


    print('Epoch: {0}\t'
          'Time {1:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'Accuracy {acc.avg:.3f}'.format(epoch,
                                          time.time()-start, 
                                          loss=losses,
                                          acc=accs))

def evaluate(test_loader, model):
    model.eval()

    # Track metrics
    accs = AverageMeter()  # accuracies

    # Evaluate in batches
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(test_loader):

        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        # Forward prop.
        scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                     words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        accs.update(accuracy, labels.size(0))

    # Print final result
    print('\n * TEST ACCURACY - %.1f per cent\n' % (accs.avg * 100))

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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(epoch, model, optimizer, pt_model, rev_label_map, filename = 'checkpoint_han.pth.tar'):
    state = {'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'pt_model': pt_model,
            'rev_label_map': rev_label_map}
    torch.save(state, filename)

