from pathlib import Path
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset
from dataset import load_FEMNIST_dataset, train_FEMNIST, test_FEMNIST

dataset_dir: Path = Path("femnist")
centralized_partition: Path = dataset_dir / 'client_data_mappings' / 'centralized'
partition_dir = dataset_dir / 'client_data_mappings' / 'fed_natural'

class Client:
    def __init__(self, dir, cid, device, epoch=10, batch=32, workers=2) -> None:
        self.cid = cid
        self.file_path = dir / str(self.cid)
        #print("data path = ",self.file_path)
        self.device = device
        self.batch_size = batch
        self.num_workers = workers
        self.epochs = epoch
        self.lr = 0.01
        self.weight_decay = 0.001
        self.train_loader: DataLoader = self._create_data_loader(name="train")
        self.test_loader: DataLoader = self._create_data_loader(name="test")
    
    def fit(self, net):
        self.init_weight = deepcopy(net.get_weights())
        net = deepcopy(net)
        net.to(self.device)
        self._train(net, train_loader=self.train_loader)
        self.final_weight = deepcopy(net.get_weights())
        return net, len(self.train_loader)

    def evaluate(self, net):
        net = deepcopy(net)
        net.to(self.device)
        loss, accuracy = self._test(net, test_loader=self.test_loader, masks=None)
        return loss, len(self.test_loader), accuracy

    def process(self,net):
        new_net, train_len = self.fit(net)
        loss, test_len, acc = self.evaluate(new_net)
        return acc, loss, train_len+test_len, self.final_weight

    def _create_data_loader(self, name: str) -> DataLoader:
        dataset = self._load_dataset(name)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False if name == 'test' else True,
        )
    
    def _load_dataset(self, name) -> Dataset:
        return load_FEMNIST_dataset(mapping=self.file_path,name=name)
    
    def _train(self, net, train_loader: DataLoader):
        return train_FEMNIST(
            net=net,
            train_loader=train_loader,
            epochs=self.epochs,
            device=self.device,
            optimizer=torch.optim.AdamW(net.parameters(),lr=self.lr,weight_decay=self.weight_decay,),
            criterion=torch.nn.CrossEntropyLoss(),
        )

    def _test(self, net, test_loader: DataLoader, masks=None):
        return test_FEMNIST(
            net=net,
            test_loader=test_loader,
            device=self.device,
            criterion=torch.nn.CrossEntropyLoss(),
            masks=masks,
        )
    
if __name__ == "__main__":
    from model import MLP
    c0 = Client(dir=partition_dir,cid=100,epoch=20)
    model = MLP(alpha=1.0)
    acc, data_len, weights = c0.process(model)
    print("Accuracy: ",acc)
    print("Data length: ",data_len)
    print("Weights")
    for w in weights:
        print(w.shape)