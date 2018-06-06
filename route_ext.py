from functools import partial


class RouteExtractor():
    def __init__(self, model, idxs):
        self.__list = [0] * len(idxs)
        self.__hooks = []
        self.__create_hooks(model, idxs)

    def __create_hooks(self, model, idxs):
        help_idxs = list(range(len(idxs)))
        for i, idx in zip(idxs, help_idxs):
            fun = partial(self.__hook_fn, idx=idx)
            hook = model[i].register_forward_hook(fun)
            self.__hooks.append(hook)

    def __hook_fn(self, module, input, output, idx):
        self.__list[idx] = output

    @property
    def routes(self):
        return self.__list

    def remove_hooks(self):
        [x.remove() for x in self.__hooks]
