# build a vocabulary for the dataset
import pickle
data_set = pickle.load(open('goal_set.p', 'rb'))
data_set.keys()
print(f'{data_set.keys()}:{[len(x) for x in data_set.values()]}')
sym_set = set()
dis_set = set()
label_set = set()
for items in (data_set['train'],data_set['test']):
    for item in items:
        dis_set.add(item['disease_tag'])
        for key,val in item['goal']['explicit_inform_slots'].items():
            sym_set.add(key)
            label_set.add(val)
        for key,val in item['goal']['implicit_inform_slots'].items():
            sym_set.add(key)
            label_set.add(val)

art_symbols = ['[PAD]','[PAD2]','[UNK]','[SEP]','[CLS]','[MASK]','[true]','[false]']
print(list(sym_set))
print(list(dis_set))
with open('vocab.txt','w') as f:
    f.write('\n'.join(art_symbols) +'\n')
    f.write('\n'.join(list(sym_set)) +'\n')
    f.write('\n')
    f.write('\n'.join(list(dis_set)))