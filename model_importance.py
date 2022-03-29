import torch.nn as nn
import torch

input_size = 10
hidden_size = 16
N = 25
seq_len = 5


class Model(nn.Module):
    def __init__(self, *args):
        super(Model, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x, hn = self.rnn(x)
        x = x[:, -1, :]
        return self.fc(x)


model = Model()

# sample_x = torch.randn(1, seq_len, input_size)
# sample_x.requires_grad = True
# predict = model(sample_x)
# positive_logist = predict[:, 1]
# (grad,) = torch.autograd.grad(positive_logist, sample_x)
# print(grad)


def compute_integrated_gradient(batch_x, model):
    batch_blank = torch.zeros_like(sample_x)

    mean_grad = 0
    n = 100

    for i in range(1, n + 1):
        x = batch_blank + i / n * (batch_x - batch_blank)
        x.requires_grad = True
        y = model(x)
        logit_positive = y[:, 1]
        (grad,) = torch.autograd.grad(logit_positive, x)
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad

    return integrated_gradients, mean_grad


# # def plot_importance(arrays, titles, output_path=None):
# #     fig, axs = plt.subplots(1, len(arrays))

# #     fig.set_figheight(7)
# #     fig.set_figwidth(10)

# #     for i, ((title1, title2), (array1, array2)) in enumerate(zip(titles, arrays)):
# #         axs[i].scatter(range(len(array1)), array1,
# #                        label=title1, s=6, marker="+")
# #         axs[i].scatter(range(len(array2)), array2,
# #                        label=title2, s=6, marker="v")
# #         axs[i].legend()
# #         axs[i].set_xlabel('Feature #')

# #     if output_path:
# #         fig.tight_layout()
# #         plt.savefig(output_path)


sample_x = torch.randn(1, seq_len, input_size)

integrated_gradient, mean_grad = compute_integrated_gradient(
    sample_x, model
)

print(integrated_gradient)

# integrated_gradient = integrated_gradient.squeeze().cpu().data.numpy()
# mean_grad = mean_grad.squeeze().cpu().data.numpy()
