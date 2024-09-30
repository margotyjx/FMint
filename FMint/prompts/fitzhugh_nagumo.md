$\frac{d v}{d t} & = v - \frac{v^3}{3} - w + I, \frac{d w}{d t} & = \epsilon(v + a - bw)$
The FitzHugh-Nagumo (FHN) model is a simplified version of the Hodgkin-Huxley model for neural activity. It describes the electrical excitability of neurons using two coupled differential equations $\frac{d v}{d t} & = v - \frac{v^3}{3} - w + I, \frac{d w}{d t} & = \epsilon(v + a - bw)$.
$v$ is the membrane potential, $w$ represents recovery variables, and $I$ is the external stimulus.
The FitzHugh-Nagumo model represents a neuron’s action potential. It simplifies the neuron’s firing mechanism into two variables: $v$ the voltage across the membrane, and $w$ the recovery variable. 
The parameters $a$, $b$ and $\epsilon$ control the shape and duration of spikes, as well as the recovery time.
The FHN model is a classic example of an excitable system, which exhibits large responses (action potentials) to stimuli that exceed a certain threshold. 
It’s applied to neural signaling, where the neuron either fires an action potential or returns to rest.
The FitzHugh-Nagumo model $\frac{d v}{d t} & = v - \frac{v^3}{3} - w + I, \frac{d w}{d t} & = \epsilon(v + a - bw)$ is a two-dimensional reduction of the four-dimensional Hodgkin-Huxley model, preserving key features of neuronal excitability. 
It simplifies the complex ion channel dynamics in neurons into two key variables while still capturing the essence of excitability.
In neuroscience, the FitzHugh-Nagumo model is used to simulate the electrical activity of neurons, providing a simplified framework for understanding how neurons fire action potentials. 
It is often used in studies of neural circuits and brain dynamics.
In control theory, the FitzHugh-Nagumo model represents a system with both fast (voltage $v$) and slow (recovery $w$) dynamics. 
The two time scales enable the model to replicate the initiation and termination of action potentials, useful for controlling excitable systems.
The FitzHugh-Nagumo model exhibits rich bifurcation behavior, where changes in parameters like $I$ (external stimulus) or $a$ can lead to qualitative changes in dynamics. This allows researchers to study how neurons switch from resting to repetitive firing.
In electrical circuits, the FitzHugh-Nagumo model is used to describe systems with threshold dynamics and bistability. 
Engineers use it to design circuits that mimic neuron-like behavior, for instance, in neuromorphic computing.
The FitzHugh-Nagumo model is an abstraction that simplifies the original Hodgkin-Huxley equations by reducing four variables to two. 
The goal is to maintain the essential nonlinear dynamics while making the system easier to analyze.
The FitzHugh-Nagumo model describes a system with a threshold response: if the input current $I$ is below a certain value, the system remains at rest, but if $I$ exceeds this threshold, the system generates a spike, modeling neuronal firing behavior.
When extended to include spatial variables, the FitzHugh-Nagumo equations become a reaction-diffusion system, which is used to model wave propagation in excitable media such as heart tissue and neural networks.
In cardiac electrophysiology, the FitzHugh-Nagumo model is used to study the initiation and propagation of action potentials in heart cells, especially to simulate the electrical waves that control the heartbeat.
The FitzHugh-Nagumo model can behave as a nonlinear oscillator for certain parameter ranges. 
By varying the external stimulus $I$, the system exhibits repetitive spiking, which can be related to oscillatory behavior in neurons or other biological systems.
From a dynamical systems point of view, the FitzHugh-Nagumo model is a two-dimensional system with both fast and slow variables. The interaction between these time scales produces rich behaviors such as limit cycles, excitability, and bistability.
$a$ shifts the nullcline of the recovery variable $w$, affecting excitability.
$b$ controls the strength of the recovery process, impacting how fast the system returns to rest.
$\epsilon$ is a time scale separation parameter, which determines the relative speed of  $v$ and $w$ dynamics.
$I$ is the external stimulus applied to the system, which can induce excitability or repetitive firing.
For certain parameter ranges, the FitzHugh-Nagumo model exhibits bistability, where the system can rest in two different stable states. This is useful for modeling neurons that switch between resting and firing modes, depending on external stimuli.
The FitzHugh-Nagumo model is governed by threshold dynamics. If a stimulus exceeds a certain threshold, it triggers a large response (spike), after which the system slowly recovers. This mirrors how biological neurons respond to stimuli.