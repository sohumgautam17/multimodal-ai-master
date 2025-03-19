import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchinfo import summary

class CNN_Encoder(nn.Module): 
    def __init__(self, embed_size):
        super().__init__()

        # Use ResNet18 which is simpler and well-established
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        
        # Add a new FC layer to get the embedding size we want
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

    def forward(self, images):
        features = self.resnet(images)
        # Flatten the feature map/image
        features = self.flatten(features)
        # Pass through our new FC layer
        features = self.fc(features)
        # Apply dropout
        features = self.dropout(features)
        return features

class LSTM_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_features, captions):
        # Embed the captions
        embeddings = self.dropout(self.embed(captions))
        
        # Append the encoder features as the first "word" in the sequence
        embeddings = torch.cat((encoder_features.unsqueeze(0), embeddings), dim=0)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embeddings)
        
        # Get predictions for each word
        outputs = self.fc(lstm_out)
        return outputs


class CNN_to_LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super().__init__()

        self.encoder = CNN_Encoder(embed_size)
        self.decoder = LSTM_Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        # Encode the images
        encoder_features = self.encoder(images)
        # Decode with the captions
        outputs = self.decoder(encoder_features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=30):
        result = []
        hidden = None  # We'll let LSTM initialize its own state
        
        with torch.no_grad():
            # Get image features from encoder
            x = self.encoder(image).unsqueeze(0)
            
            # Generate caption one word at a time
            for i in range(max_length):
                # Get LSTM output
                output, hidden = self.decoder.lstm(x, hidden)
                
                # Get predicted next word
                output = self.decoder.fc(output.squeeze(0))
                predicted_word = output.argmax(1)
                
                # Add predicted word to result
                word_idx = predicted_word.item()
                result.append(word_idx)
                
                # End if we predict the end token
                if word_idx == vocabulary.word_to_index["<EOS>"]:
                    break
                    
                # If we want to feed predicted word back as input this is the code
                # x = self.decoder.embed(predicted).unsqueeze(0)
                
        # Convert word indices to actual words and return
        return [vocabulary.index_to_word[idx] for idx in result]

def strength_test():
    # Create tensors that match the dimensions from your dataset
    image_tensor = torch.randn([32, 3, 224, 224])  # 32 images from batch
    captions = torch.randint(0, 10000, (20, 32))   # 20 words for 32 captions

    # Initialize model with smaller dimensions for quicker testing
    model = CNN_to_LSTM(embed_size=256, hidden_size=256, num_layers=1, vocab_size=10000)

    summary(model, input_data=[image_tensor, captions])

    # Forward pass
    output = model(image_tensor, captions)
    
    # Print output shape
    print("Output tensor shape:", output.shape)
    print("Test completed successfully!")

if __name__ == "__main__":
    strength_test()
