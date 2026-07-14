# Hamiltonian-Geometric Optimization: Speaker Notes

## Slide 1

Open with the high-level claim. The project is a framework, not a magic replacement for Adam. Its value is that it gives a principled language for curvature, momentum, memory, and phase-space control.

## Slide 2

Explain that we are trying to make optimization match the geometry of the problem. This is most compelling for physics-informed models, quantum control, spin systems, and tensor networks.

## Slide 3

This is the cleanest conceptual slide. Parameters become coordinates, the loss is potential energy, the metric tells us what a unit step means locally, and momentum carries state. Damping is essential because pure Hamiltonian mechanics conserves energy, while optimization should decrease loss.

## Slide 4

Be very honest here. The math tools are standard. The contribution is the assembly into an optimizer and the experiments. Say: if we do not have a direct reference for a specific design choice, we label it as our proposed modeling choice.

## Slide 5

Use this slide to answer 'how do you know it is derived like this?' We derive by taking limits or choices: flat metric, diagonal metric, no momentum, full metric. Where it is a correspondence rather than a historical derivation, say correspondence.

## Slide 6

This is a visual evidence slide. Emphasize that the method produces a trajectory in phase space, not only a scalar loss curve.

## Slide 7

For the PINN result, say this is the strongest argument for the framework: physics problems have operators, boundary conditions, and curvature. The phase-space picture makes the optimizer behavior visible.

## Slide 8

Be transparent. This builds credibility. Say that the diagonal reduction of our method behaves close to Adam, which is expected. The full framework matters more where full metric structure is available.

## Slide 9

Connect to spin chains and tensor networks here. Say the current quantum tests are small but directionally relevant: they are closer to the target future use cases than plain image classification.

## Slide 10

This is your progress slide. Use it to show the project is not just an idea. Mention that negative results were kept, including memory metric hurting in the tested quantum benchmark.

## Slide 11

This slide protects you. Audiences trust you more when you state limitations clearly. It also sets up the next slide: where the method should go next.

## Slide 12

Tensor networks are probably the strongest future direction. Stress that they already have local geometry and gauge structure, so a geometric optimizer is not artificial there.

## Slide 13

This gives a credible physics roadmap. Mention TDVP and DMRG as baselines. Do not claim we have done this yet unless asked; say this is planned integration.

## Slide 14

This answers the user's request for other higher-order optimization problems. Keep it grounded: the method is worth trying when curvature or structure matters enough to pay extra compute.

## Slide 15

End with a practical plan. The audience should leave knowing what is done, what is honest uncertainty, and what the next milestone is.

## Slide 16

This is a backup or closing slide. It directly addresses the concern about references. Use it if the audience asks what is genuinely novel.

## Slide 17

This is a backup or closing slide. It directly addresses the concern about references. Use it if the audience asks what is genuinely novel.
