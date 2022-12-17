# Probabilistic attention model

We model the attention as time-inhomogeneous discrete Markov chain. In this context, "time steps" refer to steps in the autoregressive decoder.

At each time step $i$, the probability of the attention being focused on encoder state $j$ is $\bm{\pi}^{(i)}_{j}$. Naturally, $\sum_j \bm{\pi}^{(i)}_{j} = 1$ and $\bm{\pi}_{j}$ describes a probability mass distribution.

State probability distribution is updated with the general Markov model rule $\bm{\pi}^{(i+1)} = \bm{\pi}^{(i)} P^{(i)}$, where $P_{i}$ is the state transition matrix for decoder state $i$.

Monotonicity of the attention can be ensured by limiting state transitions. Here we allow loopback transitions, moving to the next encoder state, or skipping the next encoder state. In this case, the transition matrix becomes tri-diagonal:
$$
P = \begin{pmatrix}
p_{0,0} & p_{0,1} & p_{0,2} & 0 & ... & 0\\
0 & p_{1,1} & p_{1,2} & p_{1,3} & ... & 0 \\
... & ... & ... & ... & ... & ... \\
0 & 0 & 0 & 0 & ... & p_{k,k} \\
\end{pmatrix}
$$

To ensure the right-stochasticity of transition matrix $P$, we take the column-wise softmax 

$$\lparen p^{(i)}_{j \rarr j}, p^{(i)}_{j \rarr j+1}, p^{(i)}_{j \rarr j+2} \rparen = \mathrm{softmax} \lbrack (h_j G_0(g_i), h_j G_1(g_i), h_j G_2(g_i)) \rbrack $$

$$\lparen p^{(i)}_{k-1 \rarr k-1}, p^{(i)}_{k-1 \rarr k} \rparen = \mathrm{softmax} \lbrack (h_{k-1} G_0(g_i), h_{k-1} G_1(g_i)) \rbrack $$

$$p^{(i)}_{k \rarr k} = \mathrm{softmax} \lbrack (h_k G_0(g_i)) \rbrack = 1$$

Note that the encoder output $h_k$ is not used, but in practice the input string can always be padded by an end symbol.

Encoder outputs $\bm{h}_j$, decoder hidden state $\bm{g}_i$, functions (e.g. fully connected layers with activations) $G_0, G_1, G_2$

Attention output is calculated as a sum over encoder outputs weighted by state probabilities:
$$ y^{(i)} = \sum_j \pi^{(i)}_j h_j
