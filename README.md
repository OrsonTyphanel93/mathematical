# Diffusion models

 Given a Neural Net (or other function $E: \mathbb{R}^D \to \mathbb{R}$), the distribution is given by:

$$
p(x) = \frac{e^{-E_\theta(x)}}{\int dx e^{-E_\theta(x)}} = \frac{e^{-E_\theta(x)}}{Z(\theta)}
$$

where we've defined the partition function $Z(\theta) = \int dx e^{-E_\theta(x)}$. $\theta$ are some learnable model parameters that we're going to autodiff the hell out of. To learn the generative distribution, we want to maximize the log probability. Why? Because logs make things look good, and logarithms preserve optimization. Maxing out the probability of dataset $\mathcal{D}$, and the log-probability give the same answers.

$$
p(x) = \frac{e^{-E_\theta(x)}}{\int dx e^{-E_\theta(x)}} \implies  \ln p(x) = -E_\theta(x) - \ln  \int dx e^{-E_\theta(x)}
$$

We need to know $p(x)$ to calculate the update to compute $p(x)$! One way to approach this, is iteratively sample from the current $p(x)$, compute $E(x)$, sample, compute, sample, compute, ... until convergence. In any rate, it's probably not ideal.

There is an alternative though. Notice what happens if we instead try to solve for $\nabla_x p(x)$. The normalization disappears! This leads us to define the notion of the _score_

$$
s_\theta(x) \equiv -\ln \nabla p(x) = \nabla E_\theta(x),
$$

which does not contain a pesky normalization constant. This is great, but how does it help us recover $p(x)$? Also, wouldn't optimizing this lead to some horrible Jacobian/second derivatives? Yes to the second. That's a big issue. That optimization objective also goes by the name of Fisher divergence (I think? might be getting the name wrong). Let's answer the first complaint though. It turns out, that there is an intimate relationship between the score and the distribution, which is related to the relationship between Langevin equations and Fokker-planck equations. The score $s(x)$ is like a Force that leads to a potential $p(x)$. We'll derive all of this later, but for now, let's leave the discussion here and think about a different approach to remove $Z$.

### Normalizing flow mentality

The reason $Z$ is hard to find, is because $p(x)$ is an arbitrary, complicated, unconstrained function (neural nets are universal approximators! blah blah blah...). In this small section, we're going to give a preview of the idea behind diffusion models.

Let's take a page out of the normalizing flows notebook, and try to make this problem tractable by making $p(x)$ simpler. Let's assume that $p(x) = N(x| \mu ,\Sigma)$. But wait... thats a pretty wild assumption right? Yeah, it is. It's a good thing we can take the usual physics approach though, and just find a limit where this isn't so crazy. It turns out that that's pretty easy to do actually. Let's introduce a new auxiliary parameter $t$, which we will call the "time". The function $p(x, t)$ has the following properties

1. $p(x, t \to  0)$ is something easy to understand. Let's call it $N(x|0, 1)$

2. $p(x, t \to  \infty)$ is exactly what we want, $p(x)$.

3. There exists some transition kernel $T$ such that $p(x, t + \delta t) = \int dx' T_{\delta t}(x, x', t)p(x', t)$ (this is just the [chapman-kolmogorov equation](https://en.wikipedia.org/wiki/Chapman%E2%80%93Kolmogorov_equation))

There's no loss of generality here, but you might question why make this problem harder. The reason, is that for small changes in times $\delta t$, we know that the transition kernel must become a delta function. And, since we can recover a delta function as the limit of a Gaussian with zero variance, there must some small window of time where the transition is basically Gaussian. And that's it. We'll need to iterate step (3) lots of times to get the answer that we want, but at least we've removed the normalization constant.


## Langevin, Fokker Planck, and Path integrals

1. Langevin equations are ODEs that tell you how to generate one sample from a distribution, by simulating the motion of a particle.
2. Fokker-Planck equations are a PDE that tells you how the entire distribution changes over time
3. Path integrals are functional integrals that are designed to compute conditional probabilities to go from state A to state B in some amount of time $t$


### Langevin

The Langevin equation is just an over-damped equation of motion plus some random, time-independent gaussian noise $\eta(t)$

$$
\frac{dx}{dt} + \nabla V = \eta(t), \\
p(\eta) \sim e^{-\eta^2 / 4D}
$$

In discrete form, it would look like:

$$
x_{t+1} = x_t + \epsilon F_t + \int_t^{t + \epsilon} \eta(\tau) d\tau
$$

We can convert the ODE to a probability density p(x) using the probability change of variable rule

$$
p(\eta) = p(x) \det\left|\frac{\delta  \eta}{\delta x}\right|^{-1},
$$

giving:

$$
p(x) \sim e^{-(\dot x + \nabla V)^2 / 4D} \det\left|\frac{\delta  \eta}{\delta x}\right|.
$$

There are two different ways to approach this determinant. If we make the Itô choice and discretize as

$x_{n+1} = x_n + F_n \delta t + \eta_n$, the Jacobian is:

$$
\det\left|\frac{\delta  \eta}{\delta x}\right| = 1
$$

If instead, we smear out the noise over the interval $\delta t$ instead of acting as an initial kick, we find a more complicated answer. The sketch for this case is as follows:

$$
\det\left|\frac{\delta  \eta}{\delta x}\right| = \det|\partial_t + \nabla^2 V| = \text{exp}\left( \ln (1 + \partial_t^{-1} \nabla^2 V)\right) = \frac{1}{2}\nabla^2 V
$$

(you'll need to know that the heaviside-step function is the inverse $\partial_t$, then taylor expand the logarithm)

In short, the Langevin equation tells us that the transition probability to a new state is given by a normal distribution:

$$
T_{\delta t}(x, t|x') = \mathcal{N}(x|x' + \delta t F(x, t), 2 D(x)).
$$

Notice, that this fits the form of our Gaussian transition kernel postulated earlier. That's nice, it turns out that if we know the transition kernel, we can use the Langevin equation to generate samples.

### Path integrals

Just to make things look familiar, here is the transition kernel equation written in the usual bra-ket notation:

$$
p(x, t+ \epsilon) = \int dx'' \langle x | \hat T(\epsilon) | x'' \rangle p(x'', t), \ \\ \\ \ \langle x | \hat T(\epsilon) | x'' \rangle  \sim e^{-(x_t - x_{t-1} - \epsilon F_{t-1})^2  \epsilon / 4D}
$$

This is in standard form, and so we can immediately write the path integral

$$
p(x_f, t) = \int_{x(0) = x_i}^{x(t) = x_f} \mathcal{D} x\ \ \text{exp}\left[ \frac{-1}{4D} \int_0^t(\dot x^2 + \nabla V(x))^2d\tau\right] p(x_i, 0)
$$

As is usual with path integrals, if we discretize them we get a chain of transition operators, which is equivalent to a Markov chain of conditional probabilities $q_t(x_t) = \int  \prod_{i=1}^{t} dx_{i-1} q_i(x_i | x_{i-1})$


### Fokker-Planck equations

We'll derive this in a hand-wavy manner by finding the generator function. Consider an arbitrary function $f(x)$, for which we want to find its time-dependent expectation. Just like in quantum mechanics, this can be expressed in a Schrodinger or Heisenberg representation, meaning the time dependence is either in the state or not. Let's adopt the Heisenberg approach, and consider a time-dependent probability density function $p(x, t)$ such that the expectation at any time can be computed as:

$$
\langle f \rangle = \int p(x, t) f(x) dx
$$

We will now compute this via an alternative method that makes use of the Langevin equation, thereby finding an equation that leads to the Fokker-Planck equation. Consider the standard Taylor series:

$$
df = dx_i \nabla_i f + \frac{1}{2} dx_i dx_j \nabla_i \nabla_j f + \mathcal{O}(dx^3).
$$

In order to make use of the Langevin equation, we'll need to make two assumptions about the noise. (1) impose that the second moment is proportional to the time difference between events, and (2) that events are time-independent, i.e. $\langle \eta_i \eta_j \rangle \sim \delta_{ij} dt^{>0}$. With this, we can can infer the averages:

$$
dx = -\nabla V dt + d\eta, \ \\ \ dx^2 = d \eta^2 + \mathcal{O}(dt^{>1})
$$

As a result, we can find the averaged time differential

$$
\langle \partial_t f \rangle = -\nabla f + \frac{1}{2} \langle \eta^2 \rangle \nabla^2 f
$$

We can replace the left-hand side with the previous definition for time averages using the density $p(x, t)$, leading to:

$$
\int \partial_t p(x, t) f(x) dx = \int p(x, t) \left[-\nabla f(x) + \frac{1}{2} \langle \eta^2 \rangle \nabla^2 f(x) \right] dx
$$

Let's now impose that $\langle \eta^2 \rangle = 2 D$, where $D$ is a constant known as the **diffusion coefficient**. Performing integration by parts, and noting that $p(x \to \pm \infty, t) = 0$, we find the following equation:

$$
\int f(x) \partial_t p(x, t) dx = \int f(x) \left( \nabla(p \nabla V) + D \nabla^2 p \right) dx
$$

Since this must hold for any arbitrary function $f(x)$, it implies the differential relation

$$
\partial_t p = \nabla(p \nabla V) + D \nabla^2 p
$$

which is the desired result. We've shown that for every Langevin equation, there is an equivalent Fokker-Planck PDE that describes the dynamics of the distribution p(x,t). We now have a variety of representations to make use of!

## The most important equation

Now that we know the connection between Langevin and Fokker-Planck equations, consider what happens if we set the potential energy to $V(x) = -\ln q_\theta(x)$, and the diffusion constant to $D=1$. The corresponding Langevin equation would be:

$$
x_t = x_{t-1} + \epsilon  \nabla_x  \ln q_\theta(x) + \sqrt{2\epsilon} z, \\
z \sim N(0, 1)
$$

The stationary solution $e^{-V}$ now becomes $q_\theta(x)$! We've shown we can generate samples from any distribution $q_\theta(x)$, by using the above Langevin equation.


# Diffusion models

We have all of the math background we need to derive our results now, and so for the remainder of these notes, we will dive into three main papers, chronologically:

1. Sohl-Dickstein, Jascha, et al. "Deep unsupervised learning using nonequilibrium thermodynamics." _International Conference on Machine Learning_. PMLR, 2015. https://arxiv.org/abs/1503.03585
2. Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." _Advances in Neural Information Processing Systems_ 33 (2020): 6840-6851. https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
3. Song, Yang, et al. "Score-based generative modeling through stochastic differential equations." _arXiv preprint arXiv:2011.13456_ (2020). https://arxiv.org/abs/2011.13456

The first paper lays the foundation. It gives the following formula for building diffusion models.

- Parametrize and represent some transition kernel $T_{\delta t}(x_t, t | x_{t-1}; \theta)$ via a neural network
- Pick some easy prior distribution like $p(x, t=0) = \mathcal{N}(0, 1)$ that will become our target distribution at long times $p_\text{data} = p(x, t\to\infty)$
- Minimize KL divergence of $p_\text{data}(x)$ and $p_\theta(x)$. Use importance sampling to approximate the integrals in the path integral objective we gave.
  - This is where differences appear. "Diffusion" is the surrogate distribution we use for importance sampling, hence the name!
- Generate samples using the Langevin equation plus transition kernel!




There exists a stationary solution found by setting $\partial_t p = 0$, which gives the equation $\mu p = \nabla D p => p \sim e^{\int  \mu / D}$. If we suppose there is some function such that $\mu = -\nabla V$ and if $D(x, t) = D = 2k_\text{B}T$, then we find the familiar Boltzmann distribution from physics $p_\text{eq} \sim e^{-V / k_\text{B}T}$. In other words, by cleverly setting $\mu(x, t) = -D \nabla ln q(x)$, we have produced an equation whose equilibrium solution will be the target distribution $q(x)$ for any $q(x)$.

Why is this useful? After all, we can't easily parametrize a neural net to learn an arbitrary function $p(x, t)$, without makign some limiting assumptions (i.e. variational methods or normalizing flows). Well, it's useful because there is an equivalence between Fokker-Planck equations describing the time-evolution of the probability density $p(x,t)$, and the Langevin equation -- a stochastic ODE describing the trajectory $x(t)$ of a single particle, which when ensembled produces $p(x, t)$. A neural network can deal with a single realized trajectory, since that means we need only worry about one specific coordinate $x$ at any time $t$. And so, the langevin representation is conducive to generating samples from the Fokker-planck equation. Since we can also set the equilibrium solution, it means we can also sample from any distribution we want, provided we know how to take a forward step and we sample long enough.

#### Reverse diffusion

To briefly derive the FP equation from the Langevin equation, and thereby retrieve the reverse equation, we start with the differential relation

$$
dx = f(x, t) dt + g(t)dB
$$

where $\langle dB_i dB_j\rangle^2 = \delta_{ij} dt$. The expectation of any arbitrary function $F(x, t)$ can be written in two ways:

1. $\langle F(x, t) \rangle = \int F(x) p(x, t) dx$

2. $\langle F(x, t) \rangle = \int F(x, t) p(x) dx$

These are equivalent to the two views of quantum mechanics (schrodinger and heisenberg), where either the operator is time dependent and the states are stationary (option 2), or the states are time-dependent and the operator stationary (option 1). The relation is easy in bra-ket notation $\langle  \psi(t) | \hat O | \psi(t') \rangle = \langle  \psi| e^{i \hat H t} \hat O e^{-i \hat H t'} | \psi  \rangle = \langle  \psi| \hat O(t, t') | \psi  \rangle$. We'll use these two views to write the same thing, thereby deriving a new relation.

Let's go with view (2). We first taylor $F(x(t))$ to $\mathcal{O}(dx^2)$:

$$
\left  \langle dF \right  \rangle = \left  \langle  \frac{\partial F}{\partial x} dx \right  \rangle + \frac{1}{2} \left  \langle  \frac{\partial^2 F}{\partial x_i \partial x_j} dx_i dx_j \right  \rangle
$$

Substituting the differential relation and keeping only terms to $\mathcal{O}(dt)$ we find

$$
\left  \langle  \frac{\partial F}{\partial t} \right  \rangle = \frac{\partial F}{\partial x} f(x, t) + \frac{1}{2} \frac{\partial^2 F}{\partial x^2} g^2
$$

Taking a time derivative of view (1) and setting it equal to a time derivative of view (2) with the above substituted in for $\partial_t F$, we get the integral equation:

$$
\int F(x) \partial_t p(x, t) dx = \int dx\ \ p(x) \left[\frac{\partial F}{\partial x} f(x, t) + \frac{1}{2} \frac{\partial^2 F}{\partial x^2} g^2\right]
$$

Finally, we integrate by parts to find:

$$
\int F(x) \partial_t p(x, t) dx = \int dx\ \ F(x) \left[\partial_x(-f(x, t) p(x, t)) + \partial_x^2 (g^2 p(x, t)))\right]
$$

Since this must hold for any abitrary function $F(x)$, it must hold within the integrals as well, leading to the FP equation:

$$
\partial_t p(x, t) = \partial_x(-f(x, t) p(x, t)) + \partial_x^2 (g^2 p(x, t)))
$$

What's nice about this representation, is that it also allows us to compute the conjugate solution readily, whereby we instead integrate by parts on the left hand side. This gives the _reverse FP_ equation:

$$
\partial_t p_\text{rev}(x, t) = -f(x,t)\partial_x p_\text{rev}(x, t) - g^2  \partial_x^2 p_\text{rev}(x, t)
$$

## Equivalence of score-based models and diffusion models

Consider the simplest process we can write:

$$
dx = g(t) dB.
$$

These leads to a Fokker-Planck equation $\partial_t P = g^2(t) \nabla^2 P$, which has solution $\text{exp}(-(x-x')^2 / \int g^2(t)dt)$. Let's choose $g(t) = \sigma^t$. Integrating, we get variance $\text{Var}(t) = \frac{1}{2\ln \sigma}(\sigma^{2t} - 1)$ This defines a forward process transition probability:

$$
q(x(t) | x(0)) = \mathcal{N}\left(x(t)| x(0), \frac{\sigma^{2t} - 1}{2\ln \sigma}\right )
$$

Which can be evaluated at any time $t$. Note, that as $t \to \infty$, the variance explodes. Yang refers to these types of SDEs as **Variance Exploding (VE)** solutions. Yang then puts forth the following score matching objective

$$
L = \frac{1}{2} \mathbb{E}_{x \sim q(x|x_0), x_0 \sim p_\text{data}(x_0)}
\left[
  ||s_\theta(x) - \nabla \ln q(x | x_0) ||^2
\right]
$$

This is really similar to the one given by Jascha, Ho, etc., except that those guys use $q(x_{t-1} | x_t, x_0)$. Instead, we will show that the above is equivalent to the score-matching objective (i.e. $p(x)$ instead of $q(x|x')$ in the log). And, since we know there exists a langevin that can produce samples given the score, everything is just fine. The proof is actually not too bad, and comes from the appendix of

<cite>P. Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.</cite> [link]https://arxiv.org/pdf/1907.05600.pdf

Begin with the score matching objective, for some parametrizable function $s_\theta(x)$

$$
L = \frac{1}{2} \mathbb{E}_{x \sim p(x)} \left[||s_\theta(x) - \nabla \ln p(x)||^2\right] \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x) - \nabla \ln p(x)||^2 \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx p(x) s_\theta(x) \nabla \ln p(x) + \text{const} \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx s_\theta(x) \nabla p(x) + \text{const} \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx dx' s_\theta(x) p(x')\nabla p(x|x') + \text{const} \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx dx' s_\theta(x) p(x|x')p(x')\nabla \ln p(x|x') + \text{const} \\
L = \frac{1}{2}\int dx p(x) ||s_\theta(x)||^2  - 2 \int dx dx' s_\theta(x) p(x, x')\nabla \ln p(x|x') + \text{const} \\
L = \frac{1}{2}\int dx dx' p(x, x')||s_\theta(x) - \nabla \ln p(x|x')||^2 \\
L = \frac{1}{2}\mathbb{E}_{(x, x') \sim p(x, x')} \left[||s_\theta(x) - \nabla \ln p(x|x')||^2\right]
$$

when the transition kernel is gaussian, the final and initial positions cancel out after reparametrization, leaving only the noise for the score to match on $\approx \sigma_t^2 s_\theta(x, t) = \epsilon$
