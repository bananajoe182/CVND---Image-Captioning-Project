import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_dim = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = 1
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        

    def forward(self, features, captions):
        
        batch_size = captions.shape[0]
        caption_embeds = self.word_embeddings(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), caption_embeds), 1)
        out, hidden = self.lstm(inputs)
        out = self.linear(out)
               
        return out.view(batch_size, -1, self.vocab_size)
                  
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output_list = []
        
        for i in range(max_len):
        
            out, states = self.lstm(inputs, states)
            out = self.linear(out)
            max_prob, word_idx = out.max(2)
            output_list.append(word_idx.item())
            inputs = self.word_embeddings(word_idx)                     
               
        return output_list
        
        