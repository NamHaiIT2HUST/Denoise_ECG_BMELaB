class FeatureHook:
    def __init__(self, module):
        self.output = None
        self.hook = module.register_forward_hook(self.fn)

    def fn(self, module, inputs, output):
        self.output = output.detach().cpu()

    def close(self):
        self.hook.remove()
