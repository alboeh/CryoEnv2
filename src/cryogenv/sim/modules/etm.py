import numpy as np
from typing import Any, Dict, Iterable

###################
# ETM 2 component #
###################

# ETM with 2 components (3 comp is in progress)
class ETM:
    """
    ETM: small thermal/electrical model with 1, 2 or 3 components.

    Modes
    -----
    n_comp=1: electrical (e) ↔ bath (b)
      required: C_e, G_eb, T_b, R_S, L
    n_comp=2: electrical (e) + absorber (a)
      required: C_e, C_a, G_eb, G_ea, T_b, R_S, L
    n_comp=3: electrical (e) + absorber (a) + absorber-bath (a-b)
      required: C_e, C_a, G_eb, G_ea, G_ab, T_b, R_S, L

    Rules
    -----
    - default=True  -> use built-in defaults for the chosen mode; providing
                      any parameter is NOT allowed (will raise).
    - default=False -> user must provide exactly the required parameters for the mode;
                      missing or extra keys will raise.
    """

    REQUIRED: Dict[int, tuple[str, ...]] = {
        1: ("C_e", "G_eb", "T_b", "R_S", "L", "V_B"),
        2: ("C_e", "C_a", "G_eb", "G_ea", "G_ab", "T_b", "R_S", "L", "V_B"),
    }

    # Built-in defaults per mode (example numbers; adjust to your setup)
    DEFAULTS: Dict[int, Dict[str, float]] = {
        1: dict(C_e=22e-12, G_eb=2e-7, T_b=0.1, R_S=5e-3, L=200e-9, V_B=5e-6),
        2: dict(C_e=22e-12, C_a=150e-12, G_eb=7.5e-9, G_ea=140e-12, G_ab=5e-9, T_b=15e-3, R_S=50e-3, L=400e-9, V_B=5e-6),
    }


    def __init__(self, *, default: bool = False, n_comp: int, **params: Any):
        # --- validate mode ---
        if n_comp not in (1, 2, 3):
            raise ValueError(f"n_comp must be 1, 2 or 3, got {n_comp}")
        self.n_comp = int(n_comp)

        required_keys = set(self.REQUIRED[self.n_comp])

        if default:
            # using presets – do NOT allow any overrides
            if params:
                print(f"Info: With default=True, the provided parameters {sorted(params.keys())} will be ignored. If you want to set parameters manually, use default=False or the set-function.")
            values = dict(self.DEFAULTS[self.n_comp])  # copy
        else:
            # strict user-provided parameters: must match required keys exactly
            given = set(params.keys())
            missing = required_keys - given
            extra = given - required_keys
            if missing or extra:
                msg = []
                if missing: msg.append(f"missing: {sorted(missing)}")
                if extra:   msg.append(f"unexpected: {sorted(extra)}")
                raise KeyError(f"Bad parameters for n_comp={self.n_comp}; " + ", ".join(msg))
            values = dict(params)

        # assign attributes
        for k in self.REQUIRED[self.n_comp]:
            setattr(self, k, float(values[k]))

        print(f"ETM initialized as {self.n_comp}-component model with parameters: " +
              ", ".join(f"{k}={getattr(self, k)}" for k in self.REQUIRED[self.n_comp]))
        
    def set_bias(self, V_B):
        self.V_B = V_B

###################
# ETM 3 component #
###################


# class ETM3comp:
#     def __init__(self, n_comp=3, ):
