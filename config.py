import torch

class config:
    def __init__(self):
        self.src_len = 1  # length of source
        self.tgt_len = 2  # length of target
        self.n_layers = 4  # 10  # number of Encoder of Decoder Layer
        self.drop_rate = 0.2
        self.input_size = 9 # 5 # 4
        self.hidden_size = 256

        # for data
        self.pre = "./data/"
        self.preTransform = "./dataTransform/"  
        self.preResult = "./dataResult/"
        self.last = ".xlsx"
        self.data = ["sa1", "sa2", "sa3", "sb1", "sb2", "sb3", "sc1", "sc2", "sc3"]
        self.trains = ["sb1", "sb2", "sb3"]
        self.tests = ["sb1", "sb2", "sb3"] 

        # for dataset size
        self.train_size = (1200 - self.src_len) * len(self.trains)
        self.test_size = (1200 - self.src_len) * len(self.tests)

        # for training
        self.batch = 100  # 100 # 150  # 100
        self.num_epochs = 1000
        self.num_workers = 0  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = (0.00001, 10, 0.5)  

        self.pathm = "./modelResult/transform_"

        self.patience = 7 

        # for equations





cfg = config()
