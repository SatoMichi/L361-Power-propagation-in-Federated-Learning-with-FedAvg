import os
import random
from pathlib import Path
from functools import reduce
import numpy as np
import torch
import torch.nn.functional as F
from client import Client
from model import MLP

dataset_dir: Path = Path("femnist")
centralized_partition: Path = dataset_dir / 'client_data_mappings' / 'centralized'
partition_dir = dataset_dir / 'client_data_mappings' / 'fed_natural'


class Server:
    def __init__(self,client_per_round,round,device,cids=None,client_epoch=5,alpha=1.0):
        self.device = device
        self.c_per_r = client_per_round
        self.rounds = round
        cids_list = [cid for cid in os.listdir("femnist/client_data_mappings/fed_natural") if "_" not in cid]
        if not cids:
            self.cids = random.choices(cids_list,k=self.c_per_r*self.rounds)
        else:
            self.cids = cids
        print("Client ID for this ronds are ",self.cids)
        self.client_epoch = client_epoch
        self.clients = [self.generate_client(cid,client_epoch) for cid in self.cids]
        
        self.alpha = alpha
        self.model = MLP(alpha)
        self.save_path = "model/server_"+str(self.c_per_r)+"_"+str(self.rounds)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_model(round=0)

        # for recording the model state and performance per rounds
        self.round_weights = []
        self.round_weights.append(self.model.get_weights())
        self.round_accs = []
        self.round_loss = []

    def generate_client(self,cid,epoch=5):
        return Client(dir=partition_dir,cid=cid,device=self.device,epoch=epoch)
    
    def save_model(self,round=0):
        torch.save(self.model.state_dict(), self.save_path+"/round_"+str(round)+".model")
        
    
    def process(self):
        for i in range(self.rounds):
            print("Rounds: ",i+1)
            clients = self.clients[i*self.c_per_r:(i+1)*self.c_per_r] if (i+1)*self.c_per_r<len(self.clients) else self.clients[-self.c_per_r:]
            accs = []
            losss = []
            data_lens = []
            weights = []
            for c in clients:
                print("Client ID = ",c.cid)
                acc,loss,d_len,w = c.process(self.model)
                accs.append(acc)
                losss.append(loss)
                data_lens.append(d_len)
                weights.append(w)
            new_weight = self.fedavg(accs,data_lens,weights)
            self.model = self.new_model(new_weight)
            self.save_model(round=i+1)
            self.round_weights.append(self.model.get_weights())
            loss,dlens,acc = self.evaluate()
            self.round_accs.append(acc)
            self.round_loss.append(loss)
            print("Eval_Loss: {:.4f} Eval_Acc: {:.4f}".format(loss, acc))
        pruning_accs, pruning_loss = self.evaluate_pruning()
        return self.round_loss, self.round_accs, self.round_weights, pruning_accs, pruning_loss
    
    def fedavg(self,accs,d_lens,weights):
        ratio = np.array(d_lens)
        ratio = ratio/ratio.sum()
        new_weights = [[r*w for w in weight] for r,weight in zip(ratio,weights)]
        new_weight = []
        for i in range(len(new_weights[0])):
            w_i = []
            for j in range(len(new_weights)):
                w_i.append(new_weights[j][i])
            new_weight.append(reduce(lambda x,y:x+y,w_i))
            #print(new_weight[-1].shape)
        return new_weight
    
    def new_model(self,new_weight):
        for w,layer in zip(new_weight,self.model._layers):
            layer.w = torch.nn.Parameter(w)
        return self.model
    
    def evaluate(self):
        c0 = Client(dir=centralized_partition,cid=0,epoch=20,device=self.device)
        print("Evaluating final rounds model with full test data ...")
        return c0.evaluate(self.model)
    
    def prune_by_magnitude(self,percent_to_keep, weight):
        condition = torch.abs(weight.flatten())
        how_many = int(percent_to_keep * torch.numel(condition))
        top_k = torch.topk(condition, k=how_many)
        mask = torch.zeros(condition.shape).to(self.device)
        mask[top_k.indices] = 1
        return mask.reshape(weight.shape)

    def evaluate_pruning(self):
        final_weights = self.model.get_weights()
        sparsity_levels = np.geomspace(0.01, 1.0, 20).tolist()
        acc_at_sparsity = []
        loss_at_sparsity = []
        for p_to_use in sparsity_levels:
            # Half the sparsity at output layer
            percent = [p_to_use, p_to_use, p_to_use, p_to_use, min(1.0, p_to_use * 2.0)]
            masks = []
            for p, w in zip(percent, final_weights):
                masks.append(self.prune_by_magnitude(p, w))
            #masks = torch.tensor(np.array(masks))
            c0 = Client(dir=centralized_partition,cid=0,epoch=20,device=self.device)
            test_dataloader = c0.test_loader
            loss, acc = c0._test(self.model, test_dataloader, masks=masks)
            acc_at_sparsity.append(acc)
            loss_at_sparsity.append(loss)

        return acc_at_sparsity, loss_at_sparsity


    

if __name__ == "__main__":
    s = Server(2,5)
    loss,acc,weights,p_acc,p_loss = s.process()

