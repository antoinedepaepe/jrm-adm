
import torch

class RMSpropXEstimator:
    def __init__(self, data_fidelity,
                 lr=0.001):
        """
        Initializes the optimizer settings.
        
        Args:
            inner_iter (int): Number of inner iterations.
            lr (float): Learning rate for the optimizer.
            tau (float): Additional parameter for the loss function.
            logger_ (object, optional): Logger to output loss info. If None, logging is skipped.
        """
        self.lr = lr
        self.data_fidelity = data_fidelity

    def solve(self, x,
                    thetak,
                    b,
                    x0_model,
                    tau,
                    iteration: int = 20):
        """
        Performs the inner iterations to optimize the variable x.
        
        Args:
            x_prior (torch.Tensor): Prior estimate of x.
            thetak (torch.Tensor): Parameter tensor used in the loss function.
            y (torch.Tensor): Observed data tensor.
            data_fidelity (object): Object that must have a `loss` method.
            
        Returns:
            torch.Tensor: The optimized version of x.
        """
        x_star = x0_model.detach().clone()
        x = x.detach().clone() #torch.zeros_like(x.detach().clone())
        x.requires_grad = True
        
        optimizer = torch.optim.RMSprop([x], lr=self.lr)

        for i in range(iteration):
            optimizer.zero_grad()
            
            loss = self.data_fidelity.loss(x,
                                           thetak.detach(),
                                           x_star,
                                           tau,
                                           b)
            
            print(f"Iteration {i+1}/{iteration} - Loss: {loss}")
            
            # loss already backwarded in data fidelity            
            optimizer.step()

        return x.clone().detach()

