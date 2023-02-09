import torch
import torch.nn as nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import transformers
from torch.utils.data import DataLoader
device = torch.device("cuda")
df = pd.read_csv("data.csv")

class BERTDataset():
    def __init__(self, texts, targets, max_len=64):
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(text, 
                                            None, 
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            padding="max_length", 
                                            truncation=True)
        return {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float)
        }
    
class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        # dropout for regularization
        self.bert_drop = nn.Dropout(0.3) 
        self.out = nn.Linear(768, 1)
        
    def forward(self, ids, mask, token_type_ids):
        # BERT in its default settings returns two outputs: last hidden state and output of bert pooler layer
        # we use the output of the pooler which is of the size (batch_size, hidden_size)
        # hidden size can be 768 or 1024 depending on if we are using bert base or large respectively
        _, x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # pass through dropout layer
        x = self.bert_drop(x)
        # pass through linear layer
        x = self.out(x)
        # return output
        return x

def loss_fn(outputs, targets): 
    """
    This function returns the loss.
    :param outputs: output from the model (real numbers) 
    :param targets: input targets (binary)
    """
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler): 
    """
    This is the training function which trains for one epoch 
    :param data_loader: it is the torch dataloader object 
    :param model: torch model, bert in our case
    :param optimizer: adam, sgd, etc
    :param device: can be cpu or cuda
    :param scheduler: learning rate scheduler """
    # put the model in training mode 
    model.train()
    # loop over all batches
    for d in data_loader:
        ids = d["ids"] 
        token_type_ids = d["token_type_ids"]
        mask = d["mask"] 
        targets = d["targets"]
        
        # move everything to specified device
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long) 
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        # zero-grad the optimizer
        optimizer.zero_grad()
        # pass through the model
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        # calculate loss
        loss = loss_fn(outputs, targets)
        # backward step the loss
        loss.backward()
        # step optimizer
        optimizer.step()
        # step scheduler
        scheduler.step()
        

def eval_fn(data_loader, model, device): 
    """
    this is the validation function that generates predictions on validation data
    :param data_loader: it is the torch dataloader object 
    :param model: torch model, bert in our case
    :param device: can be cpu or cuda 
    :return: output and targets
    """
    # put model in eval mode 
    model.eval()
    # initialize empty lists for targets and outputs 
    fin_targets = [] 
    fin_outputs = []
    
    with torch.no_grad():
        # this part is same as training function
        # except for the fact that there is no
        # zero_grad of optimizer and there is no loss calculation or scheduler steps.
        for d in data_loader:
            ids = d["ids"]
            token_type_ids = d["token_type_ids"] 
            mask = d["mask"]
            targets = d["targets"]
            
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long) 
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            
            # convert targets to cpu and extend the final list
            targets = targets.cpu().detach() 
            fin_targets.extend(targets.numpy().tolist())
            
            # convert outputs to cpu and extend the final list
            outputs = torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs.numpy().tolist()) 
            
    return fin_outputs, fin_targets

def train():
    # this function trains the model
    df = pd.read_csv("/Users/agustintumminello/Desktop/reviews/stock_data.csv")
    # split the data into single training and validation fold
    df_train, df_valid = model_selection.train_test_split(df, test_size=0.2, 
                                                          random_state=42, 
                                                          stratify=df.Sentiment)
    
    # reset index
    df_train = df_train.reset_index(drop=True) 
    df_valid = df_valid.reset_index(drop=True)
    
    # initialize BERTDataset from dataset.py for training dataset
    train_dataset = BERTDataset(texts=df_train.Text.values,
                                targets=df_train.Sentiment.values)
    # create training dataloader
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=8)
    # initialize BERTDataset from dataset.py
    # for validation dataset 
    valid_dataset = BERTDataset(texts=df_valid.Text.values,
                                targets=df_valid.Sentiment.values)
    
    # create validation data loader
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=4)
    
    # load model and send it to the device 
    model = BERTBaseUncased() 
    model.to(device)
    
    # calculate the number of training steps
    num_train_steps = int(len(df_train)/8*10)  
   
    # AdamW optimizer
    # AdamW is the most widely used optimizer
    # for transformer based networks
    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    # fetch a scheduler
    # you can also try using reduce lr on plateau 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    
    # start training the epochs
    best_accuracy = 0
    for epoch in range(10):
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs) 
        print(f"Accuracy Score = {accuracy}")
            
if __name__ == "__main__": 
    train()