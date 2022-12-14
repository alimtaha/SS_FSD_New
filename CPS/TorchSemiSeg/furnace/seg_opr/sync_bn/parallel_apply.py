# import threading
import queue
import torch
import torch.multiprocessing as mp
# from pathos.multiprocessing import ProcessPool as Pool
from torch.cuda._utils import _get_device_index


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.
    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices
    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    context = mp.get_context('spawn')
    # lock = threading.Lock()
    # results = {}
    # results = []
    # pool = context.Pool(len(devices))
    results_queue = queue.Queue(len(devices))
    grad_enabled = torch.is_grad_enabled()

    def _worker(module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a
                # Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            results_queue.put(output)
            # with lock:
            #     results[i] = output
        except Exception as e:
            results_queue.put(e)
            # with lock:
            #     results[i] = e

    if len(modules) > 1:
        # pool.map(_worker, [modules, inputs, kwargs_tup, devices])
        processes = [context.Process(target=_worker,
                                     args=(i, module, input, kwargs, device))
                     for i, (module, input, kwargs, device) in
                     enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for process in processes:
            process.start()
        for process in processes:
            process.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results_queue.get()
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs
