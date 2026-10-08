"""Small bounded-Lipschitz recurrent dynamics used by IRCN v0.3."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class System:
    W: np.ndarray
    U: np.ndarray
    b: np.ndarray
    tau: np.ndarray

    def rhs(self, t: float, h: np.ndarray, x: np.ndarray) -> np.ndarray:
        return (-h + np.tanh(self.W @ h + self.U @ x(t) + self.b)) / self.tau

    def rhs_node(self, i: int, h: np.ndarray, x_t: np.ndarray) -> float:
        # Only row i is evaluated; callers must supply latest arrived neighbor values.
        return float((-h[i] + np.tanh(self.W[i] @ h + self.U[i] @ x_t + self.b[i])) / self.tau[i])
