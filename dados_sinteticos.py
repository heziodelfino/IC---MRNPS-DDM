import numpy as np
from scipy.integrate import solve_ivp

def pendulum_ode(t, y, b, g, L, m):
    theta, omega = y
    # omega_dot = - (g/L) * sin(theta) - (b/(m*L*L)) * omega
    return [omega, - (g / L) * np.sin(theta) - (b / (m * L * L)) * omega]

def make_dataset(theta0=1.6, omega0=0.0, b=0.15, g=9.81, L=1.0, m=1.0,
                 t_max=20.0, N=4000, noise_sigma=0.0):
    t_eval = np.linspace(0, t_max, N)
    sol = solve_ivp(pendulum_ode, [0, t_max], [theta0, omega0],
                    args=(b, g, L, m), t_eval=t_eval, atol=1e-9, rtol=1e-8)
    theta = sol.y[0]
    omega = sol.y[1]
    if noise_sigma > 0:
        theta = theta + np.random.normal(scale=noise_sigma, size=theta.shape)
        omega = omega + np.random.normal(scale=noise_sigma, size=omega.shape)
    return t_eval, theta, omega

if __name__ == "__main__":
    # gera três amortecimentos diferentes, APENAS COMO EXEMPLO
    params = [
        {"label": "fraco", "b": 0.05},
        {"label": "medio", "b": 0.15},
        {"label": "forte", "b": 0.5},
    ]
    datasets = {}
    for p in params:
        t, th, om = make_dataset(theta0=1.8, omega0=0.0, b=p["b"], t_max=25, N=6000, noise_sigma=0.01)
        datasets[p["label"]] = (t, th, om)
    # salva em .npz
    np.savez("pendulum_synthetic.npz", **{
        f"{k}_t": v[0] for k, v in datasets.items()
    }, **{
        f"{k}_th": v[1] for k, v in datasets.items()
    }, **{
        f"{k}_om": v[2] for k, v in datasets.items()
    })
    print("Gerado pendulum_synthetic.npz com 3 datasets (fraco, medio, forte).")
