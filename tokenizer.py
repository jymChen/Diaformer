class diaTokenizer():
    vocab = {}
    ids_to_tokens = {}
    special_tokens_id = range(0,8)
    disease_tokens_id = None
    # symptom_tokens_id = None
    symptom_to_false = {}
    id_to_symptomid = {}
    tokenid_to_diseaseid = {}
    disvocab = {}
    labels_to_diseases = {}
    class_weight = None

    def __init__(self, vocab_file, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]",true_token="[true]",false_token="[false]",dis_pad_token='[PAD2]'):
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.read().split('\n\n')
            symlist = tokens[0].splitlines()
            dislist = tokens[1].splitlines()
        for index, token in enumerate(symlist):
            self.vocab[token] = index
            self.ids_to_tokens[index] = token
            index += 1
        for index, token in enumerate(dislist):
            self.disvocab[token] = index
            self.labels_to_diseases[index] = token
            index += 1
        self.unk_token_id = self.vocab[unk_token]
        self.sep_token_id = self.vocab[sep_token]
        self.pad_token_id = self.vocab[pad_token]
        self.cls_token_id = self.vocab[cls_token]
        self.mask_token_id = self.vocab[mask_token]
        self.true_token_id = self.vocab[true_token]
        self.false_token_id = self.vocab[false_token]
        self.dis_pad_token_id = self.vocab[dis_pad_token]
        self.disease_tokens_id =  range(len(self.vocab)-12,len(self.vocab))
        vocablen = len(self.vocab)
        for index in range(8,len(self.vocab)):
            false_id = index + vocablen - 8
            self.symptom_to_false[index] = false_id
            self.id_to_symptomid[false_id] = index
            self.id_to_symptomid[index] = index
        for index in range(8):
            self.id_to_symptomid[index] = index

    def convert_token_to_id(self, token):
        if token not in self.vocab:
            return self.unk_token_id
        return self.vocab[token]
    
    def __len__(self):
        return len(self.vocab)
    

    def convert_id_to_token(self,id):
        if id in self.ids_to_tokens:
            return self.ids_to_tokens[id]
        return self.unk_token_id

    def convert_disease_to_label(self,disease):
        return self.disvocab[disease]
    
    def convert_label_to_disease(self,label):
        return self.labels_to_diseases[label]