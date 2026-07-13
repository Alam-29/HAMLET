# Normal-Mode / Action-Angle Analysis

This report makes precise and numerically checks the claim that the Hamiltonian-geometric optimizer's metric exploits the loss Hessian's own symmetries via a canonical transformation to action-angle variables, and quantifies what that buys in convergence speed. See `src/normal_modes.py` for the full derivation.

## Checks

| id | kind | section | passed | max error | tolerance | detail |
|---|---|---|---|---|---|---|
| shared-eigenbasis | literal | Sec. 12 (Eq. 26) construction | PASS | 3.906e-03 | 5.624e+07 | max\|[H, g]\| commutator entry = 3.906e-03 (scale 5.624e+13) |
| plain-descent-decoupling | literal | normal-mode analysis (derived) | PASS | 2.686e-17 | 1.000e-06 | max\|theta (direct) - theta (from decoupled modes)\| after 5 steps = 2.686e-17 |
| preconditioned-descent-decoupling | literal | normal-mode analysis (derived) | PASS | 5.920e-15 | 1.000e-06 | max\|theta (direct) - theta (from decoupled modes)\| after 5 steps = 5.920e-15 |
| action-linear-hamiltonian | completed | normal-mode analysis (derived), action-angle variables | PASS | 9.831e-04 | 1.000e-02 | relative action drift over 2000 leapfrog steps (mode omega=1) = 9.831e-04 |

**shared-eigenbasis**: g = H + lambda*I commutes exactly with the loss Hessian H, so g and H share one eigenbasis -- the 'descent symmetries' the metric preconditioning exploits are literally H's own eigenvectors, not a separate or approximate structure.

**plain-descent-decoupling**: theta_{t+1} = theta_t - eta * grad L(theta_t), run directly, matches c_i,{t+1} = (1 - 2 eta lambda_i) c_i,t applied independently in each normal-mode coordinate c_i.

**preconditioned-descent-decoupling**: theta_{t+1} = theta_t - eta * g^-1 grad L(theta_t), run directly, matches c_i,{t+1} = (1 - 2 eta lambda_i/(2 lambda_i + lambda_reg)) c_i,t applied independently in each mode.

**action-linear-hamiltonian**: H_i(t) = omega_i * I_i(t) stays constant along the conservative single-mode flow (symplectic leapfrog), confirming I_i is the action variable and H_i is exactly linear in it.

## Conditioning comparison

- Raw Hessian condition number (lambda_max / lambda_min): 7.121e+08
- lambda_max = 6.824e+06, lambda_min = 0.009583
- Plain gradient descent's stability bound: eta < 1/lambda_max = 1.465e-07
- At that eta, the flattest mode contracts at rate 1.000000 per step (closer to 1 is slower; this is the mode that sets plain descent's overall speed)
- Preconditioned descent at eta = 0.9: worst-mode contraction rate = 0.999999, bounded independent of the condition number above

Normal-mode frequency range (conservative case): omega in [0.0008751, 1]
