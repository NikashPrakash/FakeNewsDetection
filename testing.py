import torch, os
import utils
from utils import config
import numpy as np
import pandas as pd
from training import _get_metrics
import clustering
from transformers import AutoTokenizer,AdamW,DistilBertModel,DistilBertTokenizer
from model import BinaryDataset, StanceDetect
from torch.utils.data import DataLoader

# checkpoint_dir = config("Cur_Hyper.checkpoint")
# filename = os.path.join(
#         checkpoint_dir, "epoch=15.checkpoint.pth.tar"
#     )


# checkpoint_dir = config("Hyper_transfer2.checkpoint")
# filename = os.path.join(
#         checkpoint_dir, "epoch=39.checkpoint.pth.tar"
#     )
# filename = os.path.join(
#         checkpoint_dir, "epoch=74.checkpoint.pth.tar"
#     )

checkpoint_dir = config("DistilBert_FineTune.checkpoint")
filename = os.path.join(
        checkpoint_dir, "epoch=4.checkpoint.pth.tar"
    )

checkpoint = torch.load(filename)

# print(f"All params: {param_set[4:]}")

# print(f"Checkpoint info : {checkpoint}\n")
if 'params' in checkpoint:
    params = checkpoint['params']
    print(f"Cur best params: {checkpoint['params']}\n")
# print(f"stats: {np.array(checkpoint['stats'])} \n")

# print(f"Cur best stats: {checkpoint['stats'][-1]}")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

_, _, x_test, y_test = clustering.cluster_then_label()
test_inputs = tokenizer(x_test.tolist(), padding=True, truncation=True, max_length=300, return_tensors='pt')

test_inputs['labels'] = y_test
test_loader = BinaryDataset(test_inputs.to('cuda'))
test_loader = DataLoader(test_loader,batch_size=64)

stats = checkpoint['stats']

criterion = torch.nn.CrossEntropyLoss()
model = StanceDetect(bert_model, 2, 0.75).to('cuda')
model.load_state_dict(checkpoint['state_dict'])

utils.make_training_plot(np.array(stats))
stats[-1] += list(_get_metrics(model, criterion, test_loader))


print(pd.DataFrame(np.array(stats[-1]).reshape(3,4), columns=['Accuracy','Loss','Recall','Precision'], index=["Val","Train","Test"]).reindex(["Train","Val","Test"]))
