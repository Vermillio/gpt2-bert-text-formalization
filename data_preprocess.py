from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from cleaner import *
import pandas as pd


def data_preprocess(dataset_path, tokenizer):
    df = pd.read_csv(dataset_path, delimiter='\t', header=None, names=['sentence', 'label'])
    sentences = df.sentence.values
    labels = df.label.values

    input_ids = []
    attention_masks = []
    sentences = clean(sentences)

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                            max_length = 64,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',  # Return pytorch tensors.
                       )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    sentences = torch.cat(sentences, dim=0)
    labels = torch.tensor(labels)

    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    dataset = TensorDataset(input_ids, attention_masks, labels, sentences)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_dataset = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_dataset,
                sampler = SequentialSampler(val_dataset), # For validation and test the order doesn't matter
                batch_size = batch_size
            )

    prediction_dataloader = DataLoader(
                test_dataset,
                sampler = SequentialSampler(prediction_data),
                batch_size = batch_size
            )

    return train_dataloader, validaton_dataloader, prediction_dataloader
