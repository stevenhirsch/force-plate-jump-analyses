# Changelog

## [0.2.0] - 2026-09-16

Fixes CMJ peak-force detection and net vertical impulse windows. This release changes
metric values, see below.

### Fixed
- `peak_force` / `frame_peak_force` now search from the start of the **braking** phase
  instead of the propulsive phase, so the correct peak is found on bimodal force curves
  where the true peak precedes the low position.
- `propulsive_net_vertical_impulse` now integrates over the full propulsive phase
  (low position to takeoff) instead of stopping at peak force.
- `braking_net_vertical_impulse` now integrates over the braking phase only
  (braking start to low position) instead of running past it into the propulsive phase.
- `braking_to_propulsive_net_vertical_impulse` redefined as braking start to takeoff,
  so the three impulse metrics are additive and each reconciles with the corresponding
  momentum change (`m * |v_min|`, `m * v_takeoff`).

### Added
- `frame_propulsive_peak_force`: a new CMJ output column giving the peak-force frame
  scoped to the propulsive phase only. Used internally so `propulsive_peakforce_rfd_*`
  metrics are unaffected by the `peak_force` fix.

### Changed
- **Breaking:** `get_peak_force_event`'s second positional parameter is renamed
  `search_start_frame` and now means "search from this frame" rather than "propulsive
  phase start." Callers passing it positionally will silently get different behavior.
  Migration: update the call to pass it as a keyword, e.g.
  `get_peak_force_event(force_series, search_start_frame=start_of_propulsive_phase)`
  reproduces the old propulsive-scoped search unchanged; pass
  `search_start_frame=start_of_braking_phase` instead to get the new, corrected
  braking-anchored search.
- `get_peak_force_event` no longer falls back to `np.argmax` when no prominent peak is
  found; it now returns `NOT_FOUND` and logs a warning. `peak_force` becomes `NaN` in
  that case instead of silently reporting a non-prominent maximum.
- `prominence` is now a keyword argument on `get_peak_force_event` (default `50` N,
  unchanged).

### Not affected
- SQJ metrics and events.
- Landing metrics.
- `propulsive_peakforce_rfd_*` (see "Added" above).
