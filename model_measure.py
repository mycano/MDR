import os
import time
import matplotlib.pyplot as plt 
import torch
import torchvision.models as models
from tqdm import tqdm
import utils


class Model_measure():
    def __init__(self, use_gpu=False) -> None:
        self.model_name = [
            "AlexNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101",
            "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19"
        ]
        self.model_num = len(self.model_name)
        print("loading model, the number of model: {}".format(self.model_num))
        self.models = [
            # the input can be (batch, 3, 224, 224)
            models.alexnet(pretrained=False),
            models.resnet18(pretrained=False),
            models.resnet34(pretrained=False),
            models.resnet50(pretrained=False),
            models.resnet101(pretrained=False),
            models.resnet152(pretrained=False),
            models.vgg11(pretrained=False),
            models.vgg13(pretrained=False),
            models.vgg16(pretrained=False),
            models.vgg19(pretrained=False)
        ]
        print("end loading model...")
        self.resolution = 224
        self.use_gpu = use_gpu
        self.can_gpu = torch.cuda.is_available()

    
    def measure_model_size(self, model_num=None, only_weight=True) -> list:
        '''
        @description: obtain the size of different model
        @param {model_num, int}  if None, obtain the size of all models, else obtain the size of model[model_num]
        @param {only_weight, boolean} if True, obtain the size of model.state_dict(), else the size of model.
        @return {model_size, list} a list of the size of model
        '''        
        save_path = "measure_model_size2322.pkl"
        model_size = []
        if model_num is not None:
            if model_num > self.model_num or model_num < 0:
                return model_size
            else:
                if only_weight:
                    torch.save(self.models[model_num].state_dict(), save_path)
                else:
                    torch.save(self.models[model_num], save_path)
                size = os.path.getsize(save_path)
                model_size.append(utils.byte2mb(size))
                os.remove(save_path)
                return model_size
        else:
            for model, model_name in zip(self.models, self.model_name):
                print("Model_measure::model_size: measure model is {}, size is ".format(model_name), end="\t")
                if only_weight:
                    torch.save(model.state_dict(), save_path)
                else:
                    torch.save(model, save_path)
                size = os.path.getsize(save_path)
                print("size: ", size)
                model_size.append(utils.byte2mb(size))
                os.remove(save_path)
        return model_size

    def measure_inference_time(self, loop_num=100, model_num=None, use_gpu=False) -> list:
        '''
        @description: measure the inference time of different models.
        @param {loop_num, int} the loop num for measuring
        @param {model_num, int} if None, obtain the size of all models, else obtain the size of model[model_num]
        @param {use_gpu, boolean}
        @return {inference, list}
        '''       
        inference_time = []
        tmp_inf_time = 0
        x = torch.rand((1, 3, 224, 224))
        if use_gpu and self.can_gpu:
            print("the inference will be used in GPU...")
            use_gpu = True
        else:
            use_gpu = False
        if model_num is not None:
            if model_num < 0 or model_num > self.model_num:
                return inference_time
            else:
                model = self.models[model_num]
                try:
                    if use_gpu:
                        model = model.cuda()
                        x = x.cuda()
                    tmp_inf_time = 0
                    for _ in tqdm(range(loop_num)):
                        start_time = time.time()
                        y = model(x)
                        end_time = time.time()
                        tmp_inf_time = tmp_inf_time + end_time - start_time
                    x = x.cpu()
                    model = model.cpu()
                    inference_time.append(tmp_inf_time/loop_num)
                except:
                    print("Due to the limited capacities, the model {} can not be placed".format(self.model_name[model_num]))
        else:
            for model, model_name in zip(self.models, self.model_name):
                tmp_inf_time = 0
                if use_gpu:
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    print("memory allocation in GPU: ", torch.cuda.memory_allocated())
                    x = torch.rand((1, 3, 224, 224))
                try:
                    if use_gpu:
                        x = x.cuda()
                        model = model.cuda()
                    for _ in tqdm(range(loop_num), desc=model_name):
                        start_time = time.time()
                        y = model(x)
                        end_time = time.time()
                        tmp_inf_time = tmp_inf_time + end_time - start_time
                    x = x.cpu()
                    model = model.cpu()
                except:
                    print("model {} can not be placeed in MX450".format(model_name))
                inference_time.append(tmp_inf_time/loop_num)

        return inference_time
    
    def measure_model_loading_time(self, loop_num=100, sleep_time=1, model_num=None, use_gpu=False, only_weight=True) -> list:
        '''
        @description: measure the model loading time.
        @param {loop_num, int} the loop num for measuring
        @param {sleep_time, float} the time for sleep before loading model, to reduce the influence of bus
        @param {model_num, int} if None, obtain the size of all models, else obtain the size of model[model_num]
        @param {only_weight, boolean} if True, obtain the size of model.state_dict(), else the size of model.
        @return {loading_time, list}
        '''        
        save_path = "measure_loading_size243242.pkl"
        loading_time = []
        tmp_load_time = None
        if use_gpu and self.can_gpu:
            print("Running in GPU")
            use_gpu = True
        else:
            use_gpu = False
        if model_num is not None:
            if model_num < 0 or model_num > self.model_num:
                return loading_time
            else:
                model = self.models[model_num]
                if only_weight:
                    torch.save(model.state_dict(), save_path)
                else:
                    torch.save(model, save_path)
                tmp_load_time = 0
                for _ in tqdm(range(loop_num), desc=self.model_name[model_num]):
                    time.sleep(sleep_time)
                    start_time = time.time()
                    if only_weight:
                        model.load_state_dict(torch.load(save_path))
                    else:
                        model = torch.load(save_path)
                    if use_gpu:
                        model = model.cuda()
                    end_time = time.time()
                    if use_gpu:
                        model = model.cpu()
                    tmp_load_time = tmp_load_time + end_time - start_time
                loading_time.append(tmp_load_time/loop_num)
                os.remove(save_path)
                return loading_time
        for model, model_name in zip(self.models, self.model_name):
            tmp_load_time = 0
            if only_weight:
                torch.save(model.state_dict(), save_path)
            else:
                torch.save(model, save_path)
            for _ in tqdm(range(loop_num), desc=model_name):
                time.sleep(sleep_time)
                start_time = time.time()
                if only_weight:
                    model.load_state_dict(torch.load(save_path))
                else:
                    model = torch.load(save_path)
                if use_gpu:
                    model = model.cuda()
                end_time = time.time()
                if use_gpu:
                    model = model.cpu()
                tmp_load_time = tmp_load_time + end_time - start_time
            loading_time.append(tmp_load_time/loop_num)
            os.remove(save_path)
        return loading_time
    
    def plot_bar_by_data(self, data, xlabel="", ylabel="", tick_label=None, save_path=None):  
        plt.bar(range(len(data)), data, tick_label=tick_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=30)
        if save_path is not None:
            plt.savefig(save_path)
            print("save the image in {}".format(save_path))
        plt.show()
    
    def plot_api(self, desc, data):
        xlabel = "Different Model"
        if desc == "LOADING":
            self.plot_bar_by_data(data, xlabel=xlabel, ylabel="Loading time (s)", tick_label=self.model_name, save_path="model_loading_time_cpu.png")
        elif desc == "LOADING_GPU":
            self.plot_bar_by_data(data, xlabel=xlabel, ylabel="Loading time on GPU (s)", tick_label=self.model_name, save_path="model_loading_time_gpu.png")
        elif desc == "INFERENCE":
            self.plot_bar_by_data(data, xlabel=xlabel, ylabel="Inference time (s)", tick_label=self.model_name, save_path="inference_time_cpu.png")
        elif desc == "INFERENCE_GPU":
            self.plot_bar_by_data(data, xlabel=xlabel, ylabel="Inference time on GPU (s)", tick_label=self.model_name, save_path="inference_time_gpu.png")
        elif desc == "MODEL_SIZE":
            self.plot_bar_by_data(data, xlabel=xlabel, ylabel="Model size (Mb)", tick_label=self.model_name, save_path="model_size.png")

    def save_data_in_csv(self, data, columns, save_path):
        import pandas as pd
        if not isinstance(columns, list):
            columns = [columns]
        df = pd.DataFrame([[s] for s in data], columns=columns, index=utils.MODEL_NAME)
        df.to_csv(save_path)
    
    def save_data_in_text(self, data, file_name):
        with open(file_name) as f:
            f.writelines(data)
    
    def save_data_api(self, desc, data):
        if desc == "LOADING":
            self.save_data_in_csv(data, ['loading time'], save_path="model_loading_cpu.csv")
        elif desc == "LOADING_GPU":
            self.save_data_in_csv(data, ['loading time gpu'], save_path="model_loading_gpu.csv")
        elif desc == "INFERENCE":
            self.save_data_in_csv(data, ['inference time'], save_path="inference_cpu.csv")
        elif desc == "INFERENCE_GPU":
            self.save_data_in_csv(data, ['inference time on gpu'], save_path="inference_gpu.csv")
        elif desc == "MODEL_SIZE":
            self.save_data_in_csv(data, ['model size'], save_path="model_size.csv")
