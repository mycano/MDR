from model_measure import Model_measure


def measure_loading_time(use_gpu=False):
    mm = Model_measure()
    data = mm.measure_model_loading_time(loop_num=10, sleep_time=0.1, only_weight=True, use_gpu=use_gpu)
    # if use_gpu and mm.can_gpu:
    #     mm.plot_api("LOADING_GPU", data)
    #     mm.save_data_api("LOADING_GPU", data)
    # else:
    #     mm.plot_api("LOADING", data)
    #     mm.save_data_api("LOADING", data)
    print("measure loading time", data)


def measure_model_size():
    mm = Model_measure()
    data = mm.measure_model_size(only_weight=True)
    # mm.plot_api("MODEL_SIZE", data)
    # mm.save_data_api("MODEL_SIZE", data)
    print("model size (MB): ", data)


def measure_inference_time(use_gpu=False):
    # data = [0.016773977279663087, 0.005521423816680908, 0.009787263870239258, 0.012956650257110595, 0.024268813133239746, 
    # 0.048040003776550294, 0.027129926681518556, 0.03086839199066162, 0.036387956142425536, 0.04488077402114868]
    mm = Model_measure()
    data = mm.measure_inference_time(model_num=None, loop_num=1, use_gpu=use_gpu)
    # if use_gpu and mm.can_gpu:
    #     mm.plot_api("INFERENCE_GPU", data)
    #     mm.save_data_api("INFERENCE_GPU", data)
    # else:
    #     mm.plot_api("INFERENCE", data)
    #     mm.save_data_api("INFERENCE", data)
    print("inference time: ", data)

if __name__ == '__main__':
    use_gpu = False
    print("Start measure.......")
    measure_inference_time(use_gpu=use_gpu)
    measure_model_size()
    measure_loading_time(use_gpu=use_gpu)