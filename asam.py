import torch
from collections import defaultdict

class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        """
        Initializes the ASAM class with the specified parameters.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use.
            model (torch.nn.Module): The model to optimize.
            rho (float, optional): The step size for the ascent step.
                Defaults to 0.5.
            eta (float, optional): The step size for the descent step.
                Defaults to 0.01.
        """
        # Store the optimizer
        self.optimizer = optimizer
        # Store the model
        self.model = model
        # Store the step size for the ascent step
        self.rho = rho
        # Store the step size for the descent step
        self.eta = eta
        # Create a defaultdict to store the state of each parameter
        self.state = defaultdict(dict)

    @torch.no_grad()  # Decorator to disable gradient calculation
    def ascent_step(self):
        """
        Performs the ascent step of the ASAM algorithm.
        """
        # Initialize list to store weight gradients
        wgrads = []
        # Iterate over named parameters of the model
        for n, p in self.model.named_parameters():
            # Skip parameters with no gradients
            if p.grad is None:
                continue
            # Get the current "eps" value for the parameter from the state
            t_w = self.state[p].get("eps")
            # If "eps" is not yet stored, create a new tensor and store it
            if t_w is None:
                t_w = torch.clone(p).detach()  # Create a new tensor as a clone of the parameter
                self.state[p]["eps"] = t_w  # Store the tensor in the state
            # If the parameter is a weight parameter
            if 'weight' in n:
                # Update the tensor to be the same as the parameter
                t_w[...] = p[...]
                # Add a positive value to the absolute values of the tensor
                t_w.abs_().add_(self.eta)
                # Multiply the gradient of the parameter by the tensor
                p.grad.mul_(t_w)
            # Append the norm of the gradient to the list of weight gradients
            wgrads.append(torch.norm(p.grad, p=2))
        # Calculate the norm of the stacked weight gradients
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        # Iterate over named parameters of the model
        for n, p in self.model.named_parameters():
            # Skip parameters with no gradients
            if p.grad is None:
                continue
            # Get the current "eps" value for the parameter from the state
            t_w = self.state[p].get("eps")
            # If the parameter is a weight parameter
            if 'weight' in n:
                # Multiply the gradient of the parameter by the tensor
                p.grad.mul_(t_w)
            # Create a new tensor with the same values as the gradient
            eps = t_w
            eps[...] = p.grad[...]
            # Multiply the new tensor by the step size divided by the weight gradient norm
            eps.mul_(self.rho / wgrad_norm)
            # Add the new tensor to the parameter to update its value
            p.add_(eps)
        # Zero the gradients of all parameters in the optimizer
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    @torch.no_grad()  # Decorator to disable gradient calculation
    def ascent_step(self):
        """
        Performs the ascent step of the SAM algorithm, which is a variant of the ASAM algorithm.
        """
        # Initialize list to store gradients
        grads = []
        # Iterate over named parameters of the model
        for n, p in self.model.named_parameters():
            # Skip parameters with no gradients
            if p.grad is None:
                continue
            # Append the norm of the gradient to the list of gradients
            grads.append(torch.norm(p.grad, p=2))
        # Calculate the norm of the stacked gradients
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        # Iterate over named parameters of the model
        for n, p in self.model.named_parameters():
            # Skip parameters with no gradients
            if p.grad is None:
                continue
            # Get the current "eps" value for the parameter from the state
            eps = self.state[p].get("eps")
            # If "eps" is not yet stored, create a new tensor and store it
            if eps is None:
                eps = torch.clone(p).detach()  # Create a new tensor as a clone of the parameter
                self.state[p]["eps"] = eps  # Store the tensor in the state
            # Update the tensor to be the same as the gradient
            eps[...] = p.grad[...]
            # Multiply the tensor by the step size divided by the gradient norm
            eps.mul_(self.rho / grad_norm)
            # Add the new tensor to the parameter to update its value
            p.add_(eps)
        # Zero the gradients of all parameters in the optimizer
        self.optimizer.zero_grad()
