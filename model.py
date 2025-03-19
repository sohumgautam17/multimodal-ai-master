import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

class CNN_Encoder(nn.Module): 
    def __init__(self, embed_size):
        super().__init__()

        # Load Pretrained Model: https://pytorch.org/vision/0.20/models.html

        #Input layer to efficient net b3 is 300 x 300 x 3 (we will preprocess in another file)
        self.efficientnet = models.efficientnet_b3(pretrained=True)

        # Replace the final fully connected layer in the CNN with a new one, adjusted for our output size
        # This output of this layer will be fed to the LSTM Decoder
        self.efficientnet.fc = nn.Linear(self.efficientnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        x = self.efficientnet(images)
        output = self.dropout(self.relu(x))
        return output

class LSTM_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)

        # Output layer is vocab_size because it will output a probability for each word in the vocabulary
        # Highest probability word will be the word that the model predicts
        self.fc = nn.Linear(input_size=embed_size, output_Size=vocab_size) 
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_features, caption):
        embeddings = self.dropout(self.embed(caption))
        # Say encoder_features is a tensor of shape (32, 256), we unsqueeze it to (1, 32, 256)
        # Then we concatenate it with the embeddings tensor of shape (1, 32, 1024)
        # This gives us a tensor of shape (2, 32, 1024)
        embeddings = torch.cat((encoder_features.unsqueeze(0), embeddings), dim=0)
        # We don't use hidden and cell state so we can use "_" to ignore them
        lstm_out, (hidden, cell) = self.lstm(embeddings)
        fc_out = self.fc(lstm_out)
        return fc_out


class CNN_to_LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super().__init__()

        self.encoder = CNN_Encoder(embed_size)
        self.decoder = LSTM_Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        # This is where we do a forward pass through both the CNN and the LSTM
        encoder_features = self.encoder(images)
        outputs = self.decoder(encoder_features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=30):
        caption = []

        with torch.no_grad():
            features = self.encoder(image).unsqueeze(0)
            states = (None, None) # Represented hidden and cell initial states of the LSTM

            for i in range(max_length):
                # Unrolling the LSTM one step at a time 
                lstm_out, (hidden, cell) = self.decoder.lstm(features, states)
                # fc_out is a probability distribution over the vocabulary
                fc_out = self.decoder.fc(lstm_out)

                # Get the word with the highest probability
                output = torch.argmax(fc_out, dim=1)
                caption.append(output.item())

                features = self.decoder.embed(output).unsqueeze(0)

                if output == vocabulary.index2word["<EOS>"]:
                    break
                
    
        return [vocabulary.index2word[i] for i in caption]
    

def strength_test():
    tensor = torch.randn(32, 64, 64, 3)
    caption = torch.randint(0, 10000, (32, 10)) # Assume 10,000 words in the vocab and 32 captions of 10 words each


    model = CNN_to_LSTM(embed_size=256, hidden_size=512, num_layers=3, vocab_size=10000)

if __name__ == "__main__":
