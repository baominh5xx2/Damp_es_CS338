class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.height = None
        self.width = None
        self.recording = False
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        if not self.recording:
            return

        activation = output

        if self.reshape_transform is not None:
            if self.height is None or self.width is None:
                return
            activation = self.reshape_transform(activation, self.height, self.width)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not self.recording:
            return

        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                if self.height is None or self.width is None:
                    return
                grad = self.reshape_transform(grad, self.height, self.width)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x, H, W):
        self.height = H // 16
        self.width = W // 16
        self.gradients = []
        self.activations = []
        self.recording = True
        try:
            if isinstance(x, list):
                return self.model.forward_last_layer(x[0], x[1])
            return self.model(x)
        finally:
            self.recording = False

    def release(self):
        for handle in self.handles:
            handle.remove()
