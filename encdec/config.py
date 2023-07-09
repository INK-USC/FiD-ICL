import ast
import argparse
import json
import os

class MultiTaskConfig(object):
    def __init__(self, filenames=None, kwargs=None):
        self.data_dir = "../data"
        self.out_dir = "checkpoints/multitask"
        self.log_file = "log.txt"
        self.train_seed = 42
        self.verbose = False
        self.debug = False

        # preprocess configs
        self.tensorize_dir = "tensorized/"
        self.n_process = 1

        # model configs
        self.model = "google/t5-v1_1-base"

        # model initialization and saving
        self.init_checkpoint = None
        self.save = False

        # data configs
        self.task = "t0"
        self.dataset = "t0"
        self.seed = 100
        self.k = 16384
        self.m = 8
        self.test_k = 16
        self.lowercase = False
        self.max_input_length = 512
        self.max_output_length = 128
        self.add_special_tokens = True
        self.fid_special_tokens = False
        self.data_weighted_sample = False

        # data formatting
        # self.add_newline = False # T5 models don't have newline ("\n") mark
        self.input_prefix = "# Input #"
        self.output_prefix = " # Output #"
        self.use_prefix_for_query_input = False
        self.use_eos_for_support_output = True

        # trainer configs
        self.do_train = False
        self.do_valid = False
        self.n_gpu = 1
        self.local_rank = -1
        self.num_training_steps = 10000
        self.warmup_steps = 600
        self.save_period = 1000
        self.log_period = 10
        self.valid_period = 1000
        self.batch_size = 8
        self.gradient_accumulation_steps = 1
        self.optimizer = "adafactor"
        self.train_with_generation_loss = False
        self.loss_avg_mode = "instance"
        self.gradient_checkpointing = False

        # eval configs
        self.do_eval = False
        self.do_eval_all = False # all checkpoints in the directory
        self.eval_batch_size = 32
        self.eval_mode = "rank_classification" # rank_classification or generation
        self.model_mode = "vanilla" # vanilla, fid, icl, ensemble
        self.varying_shots = None
        self.perturbation_mode = None

        # optimization configs
        self.lr = 1e-3
        self.max_grad_norm = 1.0
        self.weight_decay = 0.0

        self.use_tensorboard = True

        if filenames:
            for filename in filenames.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(os.getenv("CONFIG_PATH", default="run_configs"), filename)

                self.update_kwargs(json.load(open(filename)), eval=False)
        if kwargs:
            self.update_kwargs(kwargs)

    def update_kwargs(self, kwargs, eval=True):
        for (k, v) in kwargs.items():
            if eval:
                try:
                    v = ast.literal_eval(v)
                except ValueError:
                    v = v
            else:
                v = v
            if not hasattr(self, k):
                raise ValueError(f"{k} is not in the config")
            setattr(self, k, v)

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        return json.dumps(self.__dict__, indent=4, sort_keys=False)

    def save_config(self, filename):
        """
        Saves the config
        """
        with open(filename, "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

class SingleTaskConfig(MultiTaskConfig):
    def __init__(self, filenames=None, kwargs=None):
        super(SingleTaskConfig, self).__init__(filenames, kwargs)

class FiDConfig(MultiTaskConfig):
    def __init__(self, filenames=None, kwargs=None):
        super(FiDConfig, self).__init__(filenames, kwargs)

class T0PretrainConfig(MultiTaskConfig):
    def __init__(self, filenames=None, kwargs=None):
        super(T0PretrainConfig, self).__init__(filenames, kwargs)

class ICLPretrainConfig(MultiTaskConfig):
    def __init__(self, filenames=None, kwargs=None):
        super(ICLPretrainConfig, self).__init__(filenames, kwargs)
