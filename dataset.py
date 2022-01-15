from torch.utils.data import Dataset

# Dataset of Diaformer
class diaDataset(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        # input_ids = self.data_list[index].strip().split('\t')
        input_ids = self.data_list[index].split('\t')
        input_ids = tuple([[int(token_id) for token_id in input_ids[0].split()],[int(token_id) for token_id in input_ids[1].split()],int(input_ids[2])])
        return input_ids

    def __len__(self):
        return len(self.data_list)