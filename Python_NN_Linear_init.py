#packages
from torch.nn import Linear

lin = Linear(in_features = 10, out_features = 5, bias = True)

#randomized initialization weights
inp = Variable(torch.randn(1,10))
lin = Linear(in_features = 10, out_features = 5, bias = True)

#insert the randomized weights to find the optimum
lin(inp)

#access weights
lin.weights
#access bias/intercept
lin.bias

