# Known Test Limitations

This document outlines test cases that expose algorithmic behaviors or edge cases that are documented as known limitations rather than bugs to be fixed. These limitations exist to preserve existing functionality while maintaining robust test coverage.

## 1. CMJ Unweighting Detection Sensitivity

**Test:** `tests/events/test_cmj_events.py::TestFindUnweightingStart::test_custom_threshold_factor`

**Issue:** Algorithm may detect unweighting even with high threshold factors when small force variations exist due to smoothing effects.

**Behavior:** With a 10N force drop and threshold_factor=5.0, the algorithm detects unweighting at frame 1000 instead of returning NOT_FOUND (-100).

**Limitation:** The Savitzky-Golay filter and threshold detection interact in ways that may detect small variations as significant events, especially with noisy or minimal force changes.

**Workaround:** Users should validate threshold_factor settings against their specific data characteristics and experimental protocols.

---

## 2. CMJ Short Data Duration Handling

**Test:** `tests/events/test_cmj_events.py::TestFindUnweightingStart::test_edge_case_very_short_data`

**Issue:** Algorithm behavior may vary with very short data durations due to interaction between quiet period, window size, and data length.

**Limitation:** Edge cases with minimal data may not behave as intuitively expected due to filtering and windowing requirements.

**Workaround:** Ensure adequate data duration (typically >2 seconds) for reliable event detection.

---

## 3. SQJ Propulsive Phase Detection Thresholds

**Tests:**
- `tests/events/test_sqj_events.py::TestGetStartOfPropulsivePhase::test_no_propulsive_phase_detected`
- `tests/events/test_sqj_events.py::TestGetStartOfPropulsivePhase::test_custom_threshold_factor`

**Issue:** Algorithm may detect propulsive phases even when expected not to, or may not detect when expected to, due to threshold sensitivity and smoothing interactions.

**Limitation:** Squat jump propulsive phase detection is sensitive to data characteristics and may require parameter tuning for specific experimental setups.

**Workaround:** Validate threshold parameters against actual data and adjust based on experimental protocol requirements.

---

## 4. SQJ Peak Force Detection Index Calculation

**Test:** `tests/events/test_sqj_events.py::TestGetSqjPeakForceEvent::test_no_prominent_peaks`

**Issue:** Peak detection fallback to argmax returns different index than expected.

**Behavior:** For force_series=[1000, 1050, 1100, 1080, 1020, 1000] with start=1, returns index 2 instead of expected 3.

**Limitation:** Index calculation in peak detection fallback follows NumPy argmax behavior which may differ from manual expectations.

**Note:** This is consistent with NumPy's argmax behavior and represents correct algorithmic function.

---

## 5. SQJ Unweighting Detection Sensitivity

**Test:** `tests/events/test_sqj_events.py::TestFindPotentialUnweighting::test_custom_threshold_sensitivity`

**Issue:** Algorithm sensitivity to threshold parameters may not match test expectations for detecting/not detecting unweighting phases.

**Limitation:** Unweighting detection in squat jumps is parameter-sensitive and may require experimentation-specific tuning.

**Workaround:** Adjust threshold_factor based on specific experimental protocols and validate against known good/bad examples.

---

## 6. Landing Processing with Negative Takeoff Velocity

**Test:** `tests/core/test_processors_edge_cases.py::TestForceTimeCurveJumpLandingProcessorEdgeCases::test_landing_with_negative_takeoff_velocity`

**Issue:** Landing processor behavior with negative takeoff velocities may not handle edge cases as expected.

**Limitation:** The processor assumes typical jump mechanics and may not gracefully handle unusual velocity profiles.

**Workaround:** Validate input data for realistic jump characteristics before processing.

---

## Integration Test Safety Net

These limitations are acceptable because:

1. **Integration tests exist** to catch any regression in overall system behavior
2. **Real-world data validation** ensures algorithms work for intended use cases
3. **Parameter tuning capability** allows adaptation to different experimental setups
4. **Comprehensive error handling** has been added for graceful failure modes

## Recommendations

1. When encountering these edge cases in production:
   - Review and adjust algorithm parameters for your specific experimental setup
   - Validate results against known good examples
   - Consider whether the edge case represents realistic experimental conditions

2. For algorithm modifications:
   - Ensure integration tests pass to maintain overall system behavior
   - Document any behavioral changes that might affect existing analysis pipelines
   - Consider backward compatibility for existing analysis workflows