from pathlib import Path
from typing import *
import csv
from PIL import Image
from PIL.Image import Image as ImageType
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

dataset_dir: Path = Path("femnist")
data_dir: Path = dataset_dir / "data"
centralized_partition: Path = dataset_dir / 'client_data_mappings' / 'centralized'
centralized_mapping: Path = dataset_dir / 'client_data_mappings' / 'centralized' / '0'
federated_partition: Path = dataset_dir / 'client_data_mappings' / 'fed_natural'

max_train_batches_per_epoch: int = 100
max_test_batches_per_epoch: int = 100


class FEMNIST(Dataset):
    def __init__(
        self,
        mapping: Path,
        data_dir: Path = data_dir,
        name: str = 'train',
        transform: Optional[Callable[[ImageType], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
    ):
        """Function to initialise the FEMNIST dataset.

        Args:
            mapping (Path): path to the mapping folder containing the .csv files.
            data_dir (Path): path to the dataset folder. Defaults to data_dir.
            name (str): name of the dataset to load, train or test.
            transform (Optional[Callable[[ImageType], Any]], optional): transform function to be applied to the ImageType object. Defaults to None.
            target_transform (Optional[Callable[[int], Any]], optional): transform function to be applied to the label. Defaults to None.
        """
        self.data_dir = data_dir
        self.mapping = mapping
        self.name = name

        self.data: Sequence[Tuple[str, int]] = self._load_dataset()
        self.transform: Optional[Callable[[ImageType], Any]] = transform
        self.target_transform: Optional[Callable[[int], Any]] = target_transform

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """Function used by PyTorch to get a sample.

        Args:
            index (_type_): index of the sample.

        Returns:
            Tuple[Any, Any]: couple (sample, label).
        """
        sample_path, label = self.data[index]

        # Convert to the full path
        full_sample_path: Path = self.data_dir / self.name / sample_path

        img: ImageType = Image.open(full_sample_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        """Function used by PyTorch to get the length of the dataset as the number of samples.

        Returns:
            int: the length of the dataset.
        """
        return len(self.data)
    
    def _load_dataset(self) -> Sequence[Tuple[str, int]]:
        """Load the paths and labels of the partition
        Preprocess the dataset for faster future loading
        If opened for the first time.

        Raises:
            ValueError: raised if the mapping file does not exist.

        Returns:
            Sequence[Tuple[str, int]]: partition asked as a sequence of couples (path_to_file, label)
        """
        preprocessed_path: Path = (self.mapping/self.name).with_suffix(".pt")
        if preprocessed_path.exists():
            return torch.load(preprocessed_path)
        else:
            csv_path = (self.mapping/self.name).with_suffix(".csv")
            if not csv_path.exists():
                raise ValueError(f"Required files do not exist, path: {csv_path}")
            else:
                with open(csv_path, mode="r") as csv_file:
                    csv_reader = csv.reader(csv_file)
                    # Ignore header
                    next(csv_reader)

                    # Extract the samples and the labels
                    partition: Sequence[Tuple[str, int]] = [
                        (sample_path, int(label_id))
                        for _, sample_path, _, label_id in csv_reader
                    ]

                    # Save for future loading
                    torch.save(partition, preprocessed_path)
                    return partition
                

# Load with appropriate transforms
def to_tensor_transform(p: Any) -> torch.Tensor:
    """Transform the object given to a PyTorch Tensor.

    Args:
        p (Any): object to transform.

    Returns:
        torch.Tensor: resulting PyTorch Tensor
    """
    return torch.tensor(p)

def load_FEMNIST_dataset(mapping: Path, name: str) -> Dataset:
    """Function to load the FEMNIST dataset given the mapping .csv file.
    The relevant transforms are automatically applied.

    Args:
        mapping (Path): path to the mapping .csv file chosen.
        name (str): name of the dataset to load, train or test.

    Returns:
        Dataset: FEMNIST dataset object, ready-to-use.
    """
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    return FEMNIST(
        mapping=mapping,
        name=name,
        data_dir=data_dir,
        transform=transform,
        target_transform=to_tensor_transform)


def train_FEMNIST(net: Module,train_loader: DataLoader,epochs: int,device: str,optimizer,criterion: Module,) -> float:
    net.train()
    running_loss, total = 0.0, 0
    for _ in tqdm(range(epochs)):
        running_loss = 0.0
        total, batch_cnt = 0, 0
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(data), labels)
            running_loss += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            
            if batch_cnt > max_train_batches_per_epoch:
                break
            batch_cnt += 1
    total += 0.0000001
    return running_loss/total

def test_FEMNIST(net: Module,test_loader: DataLoader,device: str,criterion: Module, masks) -> Tuple[float, float]:
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader,total=min(max_test_batches_per_epoch, len(test_loader)))):
            data, labels = data.to(device), labels.to(device)
            net.to(device)
            #print(masks)
            outputs = net(data,masks)
            loss += criterion(outputs, labels).item()
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i > max_test_batches_per_epoch:
                break

    accuracy = correct / total
    return loss, accuracy


if __name__ == "__main__":
    pass