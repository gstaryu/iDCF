# -*- coding: UTF-8 -*-
"""
@Project: BIND
@File   : model.py
@IDE    : PyCharm
@Author : hjguo
@Date   : 2025/7/9 11:36
@Doc    : BIND model code, CustomizedLinear comes from:: https://github.com/uchida-takumi/CustomizedLinear/tree/master
"""
import torch
import torch.nn as nn
import math


class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):  # input: (8,58,11560), weight/mask: (1497,11560)
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.matmul(weight.t())  # (8,58,11560)@(1497,11560).t() = (8,58,1497)
        if bias is not None:
            # output += bias.expand_as(output)
            output += bias
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
            # weight = repeat(weight, 'i j-> x i j', x=grad_output.shape[0])
            # grad_input = torch.einsum('bie,bej->bij', grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
            # grad_weight=torch.einsum('bei,bej->ij', grad_output, input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        # if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):  # mask: (11560, 1497) 0/1编码
        """
        extended torch.nn module which mask connection.

        Argumens
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)  # requires_grad=False，表示这个mask本身是不可学习的

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):  # (8,58,11560)
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


class BIND(nn.Module):
    def __init__(self, gene_num, output_dim, knowledge=None):
        """
        BIND model

        :param gene_num: Input gene number
        :param output_dim: Output cell type number
        :param knowledge: Prior knowledge matrix, can be None
        """
        super(BIND, self).__init__()

        self.DNN = nn.Sequential(
            nn.Linear(gene_num, 512),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        if knowledge is not None:
            self.KSNN = nn.Sequential(
                CustomizedLinear(knowledge, bias=False),

                nn.Linear(knowledge.shape[1], 512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.ReLU(),

                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),

                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),

                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.ReLU()
            )

        # final output layer
        self.fc_expr = nn.Linear(32, output_dim)
        self.fc_comb = nn.Linear(32 * 2, output_dim)

    def forward(self, gene_expression, is_knowledge=False):
        if is_knowledge:
            dnn_output = self.DNN(gene_expression)
            ksnn_output = self.KSNN(gene_expression)

            combined = torch.cat([dnn_output, ksnn_output], dim=1)
            output = self.fc_comb(combined)
        else:
            dnn_output = self.DNN(gene_expression)
            output = self.fc_expr(dnn_output)

        return output
