# Implementation of Hadamard Multitask GP based on
# https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Hadamard_Multitask_GP_Regression.html
import torch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import Kernel, IndexKernel, MaternKernel, AdditiveKernel
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import _HomoskedasticNoiseBase
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor


class HadamardGP(ExactGP):
    """The base class for a Hadamard multitask GP regression to be used in
    conjunction with exact inference.

    Args:
        num_tasks (int): number of tasks fitted by the model
        num_kernels (int, optional): number of kernels to fit; kernels are
            combined additively
        rank (int, optional): rank of the inter-task correlation
    """

    def __init__(self, train_inputs, train_targets, likelihood, num_tasks, num_kernels=1, rank=1):
        super(HadamardGP, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = HadamardMean(ConstantMean(), num_tasks)
        self.covar_module = AdditiveKernel(*[
            HadamardKernel(MaternKernel(), num_tasks, rank)
            for _ in range(num_kernels)
        ])
        # print("Additive Kernel", self.covar_module)

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        return MultivariateNormal(mean, covar)


class HadamardMean(MultitaskMean):
    """Mean function for a Hadamard Multitask GP with one learnable constant mean per task

    Args:
        base_means (:obj:`list` or :obj:`gpytorch.means.Mean`): If a list, each mean is applied to the data.
            If a single mean (or a list containing a single mean), that mean is copied `t` times.
        num_tasks (int): Number of tasks. If base_means is a list, this should equal its length.
    """

    def __init__(self, base_means, num_tasks):
        super(HadamardMean, self).__init__(base_means, num_tasks)

    def forward(self, input):
        """
        Evaluate the mean in self.base_means corresponding to each element of
        the input data, and return as an n-vector of means
        """
        i, x = input[..., [0]], input[..., 1:]

        # Get means at x for each possible task and then gather the right one
        # for each row based on the task number i.
        means = torch.cat(
            [sub_mean(x).unsqueeze(-1) for sub_mean in self.base_means],
            dim=-1
        )

        # print(means)
        # print(means.gather(dim=-1, index=i.long()).squeeze(-1).shape)
        # which mean to take based on xi: means(x1, x2, x3, x4) -> means(xi)
        return means.gather(dim=-1, index=i.long()).squeeze(-1)


class HadamardKernel(Kernel):
    """Kernel function for a Hadamard Multitask GP of the form

    K_x(x_1, x_2) \times K_i(i_1, i_2)

    where x denotes locations (e.g., in time) and i denotes a task identifier.

    Args:
        base_kernel (:obj:`gpytorch.kernels.Kernel): the base class for the location kernel K_x,
        num_tasks (int): number of tasks denoting size of task covariance K_i
        rank (int): rank of the inter-task correlation
    """

    def __init__(self, base_kernel, num_tasks, rank):
        super(HadamardKernel, self).__init__()
        self.num_tasks = num_tasks
        self.base_kernel = base_kernel

        if rank is None:
            rank = num_tasks
        self.task_covar_module = IndexKernel(num_tasks, rank)

    def forward(self, input1, input2, **params):
        # print("input1:",input1)
        # print("input2:",input2)
        i1, x1 = input1[..., 0], input1[..., 1:]
        i2, x2 = input2[..., 0], input2[..., 1:]

        # Get input-input covariance
        covar_x = self.base_kernel(x1, x2, **params)
        # print("covar_x:", covar_x.shape)
        # Get task-task covariance
        covar_i = self.task_covar_module(i1, i2)
        # print("covar_i:", covar_i.shape)
        # Multiply the two together to get the covariance we want
        # print(covar_x.mul(covar_i).shape)
        return covar_x.mul(covar_i)


class HadamardGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    Likelihood for a Hadamard multitask GP regression. Assumes a different
    homoskedastic noise for each task i

    p(y_i \mid f_i) = f_i + \epsilon_i, \quad \epsilon_i \sim \mathcal N (0, \sigma_i^2)

    where :math:`\sigma_i^2` is the noise parameter of task i.

    .. note::
        Does not currently allow for batched training.

    :param num_tasks: The number of tasks in the multitask GP.
    :type num_tasks: int
    :param noise_prior: Prior for noise parameter :math:`\sigma^2`.
    :type noise_prior: ~gpytorch.priors.Prior, optional
    :param noise_constraint: Constraint for noise parameter :math:`\sigma^2`.
    :type noise_constraint: ~gpytorch.constraints.Interval, optional

    :var torch.Tensor noise: :math:`\sigma_i^2` parameters (noise)
    """

    def __init__(self, num_tasks, noise_prior=None, noise_constraint=None, **kwargs):
        noise_covar = HadamardHomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint, num_tasks=num_tasks
        )
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self):
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value):
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self):
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value):
        self.noise_covar.initialize(raw_noise=value)

    def __call__(self, input, *args, **kwargs):
        if not args:
            raise ValueError(
                "The first element of *args must be a list of the training"
                "inputs."
            )

        # Extract the task identifiers from the first column of the inputs
        # to pass on to the evaluation of self.noise_covar
        xi = args[0][0]
        i = xi[..., [0]].long()

        # Conditional
        if torch.is_tensor(input):
            return super().__call__(input, i=i, *args, **kwargs)
        # Marginal
        elif isinstance(input, MultivariateNormal):
            return self.marginal(input, i=i, *args, **kwargs)
        # Error
        else:
            raise RuntimeError(
                "Likelihoods expects a MultivariateNormal or Normal input to make marginal predictions, or a "
                "torch.Tensor for conditional predictions. Got a {}".format(input.__class__.__name__)
            )


class HadamardHomoskedasticNoise(_HomoskedasticNoiseBase):
    r"""
    Noise for a Hadamard multitask GP regression with a different homoskedastic
    noise for each task i:
    """

    def __init__(self, noise_prior=None, noise_constraint=None, num_tasks=1):
        super().__init__(noise_prior, noise_constraint, torch.Size(), num_tasks)

    def forward(self, *params, shape=None, noise=None, i=None, **kwargs):
        # Note: removed batching and additional checks/logic for simplicity

        # For each observation, pick the noise indicated by i
        noise = self.noise
        noise_diag = noise.expand(shape[0], len(noise)).contiguous()
        noise_diag = noise_diag.gather(-1, i).squeeze(-1)
        return DiagLazyTensor(noise_diag)