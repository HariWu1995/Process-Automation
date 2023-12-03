import sys
sys.path.append("..")

import string
import argparse

# pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
# For more version: https://pytorch.org/
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from crnn_utils.utils import AttnLabelConverter
from crnn_utils.model import Model
from crnn_utils.dataset import RawDataset, AlignCollate
from crnn_utils.parameter import Parameter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class CRNN(object):
    
    def __init__(self):
        # self.opt = opt
        self.opt = Parameter()
        characters = string.printable[:-6]    
        self.converter = AttnLabelConverter(characters)
        self.opt.num_class = len(self.converter.character)

        self._generate()

    def _generate(self):
        self.model = Model(self.opt)
        self.model = nn.DataParallel(self.model).to(device)
        self.model.load_state_dict(torch.load(self.opt.model, map_location=device))
        self.model.eval()
        print("\n\n\n[CRNN] Model is loaded\n\n\n")

    def eval(self, image):
        self.loader = DataLoader(
            dataset=RawDataset(root='', opt=self.opt, image=image), 
            batch_size=self.opt.batch_size,
            num_workers=0,
            collate_fn=AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD), 
            pin_memory=False
        )
        
        with torch.no_grad():
            for image_tensors, image_path_list in self.loader:
                image = image_tensors.to(device)
                batch_size = image_tensors.size(0)
                length_for_pred = torch.IntTensor([self.opt.batch_max_length]*batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length+1).fill_(0).to(device)

                preds = self.model(image, text_for_pred, is_train=False)
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)          
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)
                
                img_name, pred, pred_max_prob = image_path_list[0], preds_str[0], preds_max_prob[0]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  
                pred_max_prob = pred_max_prob[:pred_EOS]
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                return pred, confidence_score

