import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image

# This object is to pad captions to ensure they are all the same length
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        captions = [item[1] for item in batch]
        padded_captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)

        return imgs, padded_captions

def caption_image(model, image, vocabulary, max_length=30):
    """
    Generate a caption for a single image using a trained model.
    Args:
        model: The CNN_to_LSTM model
        image: A single image tensor (1, 3, 224, 224)
        vocabulary: The Vocabulary object
        max_length: Max caption length

    Returns:
        List of predicted words
    """
    model.eval()
    result = []

    with torch.no_grad():
        encoder_features = model.encoder(image)

        word_idx = vocabulary.word_to_index["< SOS >"]
        result.append(word_idx)

        inputs = model.decoder.embed(torch.tensor([word_idx], device=image.device)).unsqueeze(0)
        x = torch.cat((encoder_features.unsqueeze(0), inputs), dim=0)

        states = None

        for i in range(max_length - 1):
            if i == 0:
                lstm_out, states = model.decoder.lstm(x)
            else:
                lstm_out, states = model.decoder.lstm(inputs, states)

            output = model.decoder.fc(lstm_out[-1])
            predicted_idx = output.argmax(1).item()
            result.append(predicted_idx)

            if predicted_idx == vocabulary.word_to_index["<EOS>"]:
                break

            inputs = model.decoder.embed(torch.tensor([predicted_idx], device=image.device)).unsqueeze(0)

    # Clean up output
    special_tokens = {"<PAD>", "<UNK>", "< SOS >", "<EOS>"}
    return [vocabulary.index_to_word[idx] for idx in result 
            if vocabulary.index_to_word[idx] not in special_tokens]
