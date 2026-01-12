# belief-agent
An explicit uncertainty-aware cognitive agent that models latent belief states and performs inference given incomplete information. 

## About

**Uncertainty-Aware Multi-Agent Inference**

This project explores how multiple agents can infer a hidden goal under noisy, ambiguous evidence while explicitly representing **belief**, **uncertainty**, and **trust in communication**.

Inspired by Bayesian cognition, predictive processing, and human collaborative reasoning, the system studies when communication is beneficial, when it isn't, and how confidence-weighted messaging stabilizes inference.

## Project Structure

This project is organized into a modular system:

- **Branch A — Belief Inference Under Uncertainty**  
  Explicit belief representations, entropy-based uncertainty, and communication between agents.

- **Branch B — Associative Memory**  
  Memory systems that store and retrieve belief–outcome associations with attractor-like dynamics.

- **Branch C — Information Integration Across Time**  
  Intend to explore temporal stability, recovery from perturbation, and integration vs fragmentation.

- **Branch D — Uncertainty-Weighted Learning**  
  Intend to explore confidence-modulated updates and comparisons to standard error-driven learning.

- **Branch E — Self-Modeling Agents**  
  Intend to produce agents that represent uncertainty about their own reliability and adjust behavior accordingly.


Current status:
- Branch A: Belief inference under uncertainty (complete)
- Branch B: Associative memory (in progress)

## Branch A — Belief Inference Under Uncertainty
**When agents have asymmetric reliability, unidirectional, confidence-weighted communication, group accuracy is significantly improved and misinformation cascades are prevented.**

# Each agent maintains a probability distribution over three possible goals (A, B, C).
At every timestep, agents:
1. Receive private evidence (sensor observations or language constraints)
2. Update beliefs via Bayesian inference
3. Communicate a structured message derived from their belief
4. Fuse incoming messages weighted by the sender’s confidence (precision)

Uncertainty is explicitly represented as entropy and modulates how much agents trust one another.

This setup serves as a parallel to human cognition and group reasoning:

- Strong perceptual evidence (sensor agent) vs. weak, ambiguous testimony (language agent)
- Overconfident early claims cause belief collapse
- Herding and agreement without correctness
- Trust calibration based on confidence and past reliability

The project demonstrates that naive symmetric communication can degrade collective performance — a failure mode observed in human groups, social media dynamics, and multi-agent systems.

I evaluated under three main communication types:

• No communication  
• Bidirectional communication (both agents exchange messages)  
• Unidirectional communication (Sensor → Language only)

Each condition is tested across increasing noise levels in private evidence.

###Noise Sweep Results

*Sensor Agent Correctness*

<img width="498" height="386" alt="sensmodelnoisesweep" src="https://github.com/user-attachments/assets/ef56ec8e-549d-4228-8437-aea72c1b05bb" />


*Language Agent Correctness*

<img width="510" height="394" alt="langmodelnoisesweep" src="https://github.com/user-attachments/assets/fcf447cb-2db5-4b0c-bf1e-a663dd43a28d" />


*Simultaneous Correctness*

<img width="498" height="389" alt="bothcorrectnoisesweep" src="https://github.com/user-attachments/assets/9a8a7e2b-b2ae-434f-b779-df8d63200106" />


*Frequency of Agent Agreement on Conclusion*

<img width="486" height="375" alt="agreementnoisesweep" src="https://github.com/user-attachments/assets/5154f848-4189-4b75-b719-f55fa449150a" />

*Further details, results, and analysis are documented inline in code and experiment logs.
A full README will be written once all branches are complete.*

