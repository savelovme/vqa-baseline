from nltk.tokenize import RegexpTokenizer
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot

def text2tokens (text, tokenizer=RegexpTokenizer(r"\w+'\w+|\w+")):
    return [tok for tok in tokenizer.tokenize(text) ]
    
def text2idxs(text, token2idx):
    return [token2idx[token] if token in token2idx else token2idx[UNK] for token in text2tokens(text)]

def imgs2tensor(images):
    img_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return torch.stack([img_preprocess(img) for img in images])

def ques2tensor(questions):
    return pad_sequence([torch.LongTensor(text2idxs(q, q_token2idx)) for q in questions], batch_first=True, padding_value=q_token2idx[PAD])

def ans2tensor(answers):
    return torch.stack([one_hot(torch.LongTensor(text2idxs(a, a_token2idx)), num_classes=len(a_tokens)).double().mean(dim=0) for a in answers])

def cross_entropy(logits, target):
    exp_sums = torch.exp(logits).sum(dim=-1)     
    return (torch.log(exp_sums) - (logits*target).sum(dim=-1)).mean()

def save_checkpoint(save_path, model, optimizer, train_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'train_loss_list': train_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['train_loss_list'], state_dict['global_steps_list']
