$$
\begin{array}{l}
\hline
\textbf{Algorithm 1: Main Training Loop} \\
\hline
\textbf{Input: } \text{Network weights } \theta, \text{ LRs } \alpha, \text{ inner objective } \ell, \text{ meta objective } \mathcal{L}, \text{ learning rate for } \alpha: \eta \\
\hline
1: \quad j \leftarrow 0, \; \mathcal{R} \leftarrow \{\}, \; \text{DD}_\mathcal{R} \leftarrow \{\} \quad \triangleright \text{ Initialise Replay Buffer and Distilled Data Buffer} \\
2: \quad \textbf{for } t := 1 \textbf{ to } T \textbf{ do} \\
3: \quad \quad \textbf{for } ep := 1 \textbf{ to } \text{num\_epochs} \textbf{ do} \\
4: \quad \quad \quad \textbf{for } \text{batch } b \textbf{ in } (X^t, Y^t) \sim \mathcal{D}_t \textbf{ do} \\
5: \quad \quad \quad \quad k \leftarrow \text{sizeof}(b) \\
6: \quad \quad \quad \quad b_m \leftarrow \text{Sample}(\text{DD}_\mathcal{R}) \cup b \\
7: \quad \quad \quad \quad \textbf{for } n = 0 \textbf{ to } k - 1 \textbf{ do} \\
8: \quad \quad \quad \quad \quad \text{Push } b[k'] \text{ to } \mathcal{R} \text{ with reservoir sampling} \\
9: \quad \quad \quad \quad \quad \theta^j_{k'+1} \leftarrow \theta^j_{k'} - \alpha^j \cdot \nabla_{\theta^j_{k'}} \ell \\
10: \quad \quad \quad \quad \textbf{end for} \\
11: \quad \quad \quad \quad \alpha^{j+1} \leftarrow \alpha^j - \eta \nabla_{\alpha^j} \mathcal{L}_t(\theta^j_k, b_m) \quad \text{(a)} \\
12: \quad \quad \quad \quad \theta^{j+1}_0 \leftarrow \theta^j_0 - \max(0, \alpha^{j+1}) \cdot \nabla_{\theta^j_0} \mathcal{L}_t(\theta^j_k, b_m) \quad \text{(b)} \\
13: \quad \quad \quad \quad j \leftarrow j + 1 \\
14: \quad \quad \quad \textbf{end for} \\
15: \quad \quad \quad \textbf{if } ep == \text{num\_epochs} \textbf{ then} \\
16: \quad \quad \quad \quad \text{save } \theta^j_0 \text{ as } \theta_t \\
17: \quad \quad \quad \textbf{end if} \\
18: \quad \quad \textbf{end for} \\
19: \quad \quad \text{Generate expert trajectory for current task and save last 30 trajectories} \\
20: \quad \quad \text{Using } \textbf{Algorithm 2} \text{ to update old tasks with trajectory shift} \\
21: \quad \quad \text{Using } \textbf{Algorithm 3} \text{ to distill current task data} \\
22: \quad \textbf{end for} \\
\hline
\end{array}
$$

---

$$
\begin{array}{l}
\hline
\textbf{Algorithm 2: Update Old Task Distilled Data with Trajectory Shift} \\
\hline
\textbf{Input: } \text{Old distilled data } \mathcal{D}_{syn}^{old}, \text{ current initial weights } \theta_{new}, \text{ old initial weights } \theta_{old}, \\
\quad \quad \quad \text{ old expert trajectories } \mathcal{T} = \{\tau_i\}_{i=1}^M, \text{ meta learning rate } \eta \\
\hline
1: \quad \Delta\theta \leftarrow \eta \cdot (\theta_{new} - \theta_{old}) \quad \triangleright \text{ Compute trajectory shift} \\
2: \quad \mathcal{T}_{shifted} \leftarrow \{\} \quad \triangleright \text{ Apply shift to all expert trajectories} \\
3: \quad \textbf{for each } \text{trajectory } \tau = \{\theta_0^*, \theta_1^*, \ldots, \theta_T^*\} \textbf{ in } \mathcal{T} \textbf{ do} \\
4: \quad \quad \tau_{shifted} \leftarrow \{\theta_0^* + \Delta\theta, \theta_1^* + \Delta\theta, \ldots, \theta_T^* + \Delta\theta\} \\
5: \quad \quad \mathcal{T}_{shifted} \leftarrow \mathcal{T}_{shifted} \cup \{\tau_{shifted}\} \\
6: \quad \textbf{end for} \\
7: \quad \mathcal{D}_{syn} \leftarrow \mathcal{D}_{syn}^{old} \quad \triangleright \text{ Initialize from old distilled data} \\
8: \quad \alpha \leftarrow \alpha_0 \quad \triangleright \text{ trainable learning rate} \\
9: \quad \textbf{for each } \text{distillation step} \textbf{ do} \\
10: \quad \quad \text{Sample expert trajectory: } \tau^* \sim \{\tau_i^*\} \text{ with } \tau^* = \{\theta_t^*\}_{t=0}^T \\
11: \quad \quad \text{Choose random start epoch} \\
12: \quad \quad \text{Initialize student network with expert params: } \hat{\theta}_t := \theta_t^* \\
13: \quad \quad \textbf{for } n = 0 \textbf{ to } N - 1 \textbf{ do} \\
14: \quad \quad \quad \text{Sample a mini-batch of distilled images: } b_{t+n} \sim \mathcal{D}_{syn} \\
15: \quad \quad \quad \hat{\theta}_{t+n+1} = \hat{\theta}_{t+n} - \alpha \nabla \ell(\mathcal{A}(b_{t+n}); \hat{\theta}_{t+n}) \\
16: \quad \quad \textbf{end for} \\
17: \quad \quad \text{Compute loss between ending student and expert params} \\
18: \quad \quad \text{Update } \mathcal{D}_{syn} \text{ and } \alpha \text{ with respect to } \mathcal{L} \\
19: \quad \textbf{end for} \\
\hline
\textbf{Output: } \text{Updated distilled data } \mathcal{D}_{syn}^{updated} \\
\hline
\end{array}
$$

---

$$
\begin{array}{l}
\hline
\textbf{Algorithm 3: Distill Current Task Data} \\
\hline
\textbf{Input: } \text{Current task data } (X^t, Y^t), \text{ current initial weights } \theta_t, \text{ expert trajectories } \mathcal{T}_t \\
\hline
1: \quad \text{Initialize distilled data } \mathcal{D}_{syn} \sim \mathcal{D}_{real} \\
2: \quad \text{Initialize trainable learning rate } \alpha := \alpha_0 \text{ for applying } \mathcal{D}_{syn} \\
3: \quad \textbf{for each } \text{distillation step} \textbf{ do} \\
4: \quad \quad \text{Sample expert trajectory: } \tau^* \sim \{\tau_i^*\} \text{ with } \tau^* = \{\theta_t^*\}_{t=0}^T \\
5: \quad \quad \text{Choose random start epoch} \\
6: \quad \quad \text{Initialize student network with expert params: } \hat{\theta}_t := \theta_t^* \\
7: \quad \quad \textbf{for } n = 0 \textbf{ to } N - 1 \textbf{ do} \\
8: \quad \quad \quad \text{Sample a mini-batch of distilled images: } b_{t+n} \sim \mathcal{D}_{syn} \\
9: \quad \quad \quad \hat{\theta}_{t+n+1} = \hat{\theta}_{t+n} - \alpha \nabla \ell(\mathcal{A}(b_{t+n}); \hat{\theta}_{t+n}) \\
10: \quad \quad \textbf{end for} \\
11: \quad \quad \text{Compute loss between ending student and expert params} \\
12: \quad \quad \text{Update } \mathcal{D}_{syn} \text{ and } \alpha \text{ with respect to } \mathcal{L} \\
13: \quad \textbf{end for} \\
\hline
\textbf{Output: } \text{Distilled data } \mathcal{D}_{syn} \\
\hline
\end{array}
$$