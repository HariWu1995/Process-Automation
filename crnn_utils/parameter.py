import os
import sys
import string


dir_path, file_path = os.path.realpath(__file__).rsplit("\\", 1)
print(dir_path, '-->', file_path)
sys.path.insert(1, dir_path)

mother_dir, _ = os.path.realpath(dir_path).rsplit("\\", 1)
sys.path.insert(1, mother_dir)


class Parameter(object):

    _defaults = {
        "model_path": os.path.join(dir_path, 'models\\TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name `{}`".format(n)

    def __init__(self, **kwargs):

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.workers = 0
        self.batch_size = 1
        self.model = self.model_path

        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.character = string.printable[:-6]
        self.num_class = len(self.character)
        self.PAD = False

        self.Transformation = 'TPS'
        self.FeatureExtraction ='ResNet'
        self.SequenceModeling ='BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256