import torch

def train_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]

    return [inputs, bboxes, labels]


def eval_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    images = [batch[b][3] for b in range(batch_size)]

    return [inputs, bboxes, labels, images]


def test_collate(batch):
    batch_size = len(batch)
    for b in range(batch_size): 
        inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
        images = [batch[b][1] for b in range(batch_size)]

    return [inputs, images]
