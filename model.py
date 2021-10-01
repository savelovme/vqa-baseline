import torch
from torch import nn
from torchvision.models import googlenet

class VQA_baseline(nn.Module):
    def __init__(self, q_vocab_size, emb_dim, a_vocab_size):
        super(VQA_baseline, self).__init__()
      
        self.cnn = googlenet(pretrained=True)
        self.emb = nn.Embedding(num_embeddings=q_vocab_size, embedding_dim=emb_dim)
        
        total_num_features = self.cnn.fc.out_features + emb_dim
        self.bn = nn.BatchNorm1d(total_num_features, dtype=torch.double)
        self.lin = nn.Linear(in_features=total_num_features, out_features=a_vocab_size,dtype=torch.double)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, imgs_tensor, ques_tensor):

        with torch.no_grad():
            img_features = self.cnn(imgs_tensor).double()
        text_features = self.emb(ques_tensor).mean(dim=1).double()

        comb_features = torch.cat((img_features, text_features), dim=1)

        output = self.lin(self.bn(comb_features))
        return output

    def predict_probas(self, imgs_tensor, ques_tensor):
        logits = self.forward(imgs_tensor, ques_tensor)
        return self.sm(logits)
