def _parse_custom_sigmas(sigmas_string, interpolate_to_steps):
    """Parse positive, descending sigma values and log-linearly interpolate."""
    raw = [part.strip() for part in sigmas_string.replace("\n", ",").split(",")]
    raw = [part for part in raw if part]
    if not raw:
        return None

    try:
        values = [float(part) for part in raw]
    except ValueError as exc:
        raise ValueError(f"Custom Sigmas: invalid number: {exc}") from exc

    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Custom Sigmas: values must be finite and non-negative.")

    # Permit an explicitly supplied terminal zero; interpolation is performed
    # over positive values and a terminal zero is appended afterward.
    if values[-1] == 0:
        values = values[:-1]
    if len(values) < 2:
        raise ValueError("Custom Sigmas: enter at least two positive sigma values.")
    if any(value <= 0 for value in values):
        raise ValueError("Custom Sigmas: zero is only permitted as the final value.")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("Custom Sigmas: values must be strictly decreasing.")

    target_steps = int(interpolate_to_steps)
    if target_steps < 1:
        raise ValueError("Custom Sigmas: interpolate_to_steps must be at least 1.")

    if len(values) != target_steps:
        source_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float64)
        target_x = np.linspace(0.0, 1.0, num=target_steps, dtype=np.float64)
        log_values = np.log(np.asarray(values, dtype=np.float64))
        values = np.exp(np.interp(target_x, source_x, log_values)).tolist()

    values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)
