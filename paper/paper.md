---
title: 'JumpMetrics: A Python package computing countermovement and squat jump events and metrics'
tags:
  - Python
  - Biomechanics
  - Force Plates
  - Vertical Jump
  - Movement Assessment
authors:
  - name: Steven Mark Hirsch
    orcid: 0000-0002-4394-1922
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Faculty of Kinesiology and Physical Education, University of Toronto, Canada
   index: 1
 - name: Tonal Strength Institute, San Francisco, United States of America
   index: 2
date: 16 July 2024
bibliography: paper.bib
---

# Summary

Researchers and practitioners commonly use countermovement jumps and squat jumps to evaluate people's capacity `[mcmahon:2017]`, determine potential risk for injury `[bird:2022]`, or assess "readiness" for training `[watkins:2017]`. Commonly, these jumps are performed on force plates to collect the relevant time series data to derive these insights. After collecting raw force data, various events are detected and metrics computed in order to derive the various insights the exercise professional wishes to glean from the individual. However, despite there being a necessity to use algorithms to consistently detect events and compute metrics from this time series data, there are currently no free, open-source alternatives for both researchers and practitioners to leverage to ensure the data they are computing are consistent between groups.

# Statement of need

`JumpMetrics` is a free, open source Python package for computing countermovement jump and squat jump events and metrics. The API for `JumpMetrics` was designed to be modular and easy-to-use so that the takeoff and landing phases of each jump can be examined individually or together depending on the needs of the user. Furthermore, `JumpMetrics` provides various other helper functions to prepare the data for detecting events and computing metrics. These involve functions to crop and low-pass filter data if both steps are appropriate for the particular analysis a researcher or practitioner is performing.

`JumpMetrics` was designed as a free and easy-to-use tool for both researchers and practiitoners in professional settings to process single jumps, or batch processing several jumps, in a consistent and open manner. Beyond making analyses with force plates more accessible to both researchers and practitioners, the goal of `JumpMetrics` is to ensure that the events and metrics are both easy to compute and reproducible. `JumpMetrics` can also be leveraged by undergraduate or graduate students to assist them with sport science-related projects.

# How the Library Works

## Event Detections and Metrics
`JumpMetrics` computes various events and metrics for both the takeoff and landing phase. The events (and thus metrics) are slightly different between the jump variations given the differences in their movement executions.
### Takeoff Phase
#### Countermovement Jumps
The takeoff phase includes both the "lowering" (sometimes referred to as the eccentric) and "ascending" (sometimes referred to as the concentric) phases of the jump. The events computed in this package correspond to the phases of the countermovement jump highlighted in `@mcmahon:2018`. These involve the start of the "unweighting" phase, the start of the braking phase, the start of the propulsive phase, and the frame corresponding to the peak force. `JumpMetrics` computes the start of the unweighting phase in the same manner as outlined in `owen:2014` whereby the first frame of force data that exceeds 5 times the standard deviation of the force data (default value, this is a tuneable parameter depending on the data collection parameters) during quiet standing defines the start of the unweighting phase. The start of the braking phase is defined as the frame corresponding to the minimum downward movement velocity (of the individual's estimated center of mass). The start of the propulsive phase is defined as the frame corresponding to the minimum downward displacement (of the individual's estimated center of mass). The peak force event is captured using `find_peaks` from the `scipy` package and looks for a "peak" in the force series.

![Example countermovement jump force-time trace with events detected during the takeoff phase.](/analyses/study_1/figures/F02/CTRL1/literature_cutoff/force.png)
Figure 1. Example countermovement jump force-time trace with events detected during the takeoff phase.


The rate of force development, net vertical impulse, and average force are computed between all events detected during the countermovement jump. Additionally, metrics such as the jump height (based on the net vertical impulse and using the impulse-momentum relationship as well as the velocity at the final frame of data before takeoff), takeoff velocity, movement time, unweighting time, braking time, propulsive time, lowering displacement are also all computed. A full list of the metrics exported for countermovement jumps are presented in Table 1.

#### Squat Jumps
Given that the squat jump is intentionally performed with a pause to remove the continuous countermovement, the only events computed are the start of the propulsive phase and the peak force event. The start of the propulsive phase is computed by examining the first frame of data that exceeded 5 times the standard deviation of the force data (default value, this is a tuneable parameter depending on the data collection parameters) during the squat phase. The peak force event is computed similarly to the countermovement jumps whereby `find_peaks` from the `scipy` package and looks for a "peak" in the force series.
![Example squat jump force-time trace during the takeoff phase.](/analyses/study_3/figures/SQT/P02/3_5/literature_cutoff/force.png)
Figure 2. Example squat jump force-time trace during the takeoff phase.

Although there is not supposed to be any countermovement during a squat jump, depending on the instructions and guidance provided to the participant, as well as their general movement behaviours, it is possible that there is a minor countermovement that would negate the trial from being a true squat jump. `JumpMetrics` specifically looks for, and flags (with a warning and an estimated frame in the metrics output), this motion to make the user aware of this potential flaw in the squat jump trial.
![Example squat jump force-time trace with an inappropriate countermovement detected during the takeoff phase.](/analyses/study_3/figures/SQT/P02/1_2/literature_cutoff/force.png)
Figure 3. Example squat jump force-time trace with an inappropriate countermovement detected during the takeoff phase.

The rate of force development, net vertical impulse, and average force are computed between all events detected during the countermovement jump. Additionally, metrics such as the jump height (based on the net vertical impulse and using the impulse-momentum relationship as well as the velocity at the final frame of data before takeoff), takeoff velocity, movement time, and propulsive time are also all computed. A full list of the metrics exported for squat jumps are presented in Table 1.

### Landing Phase
The landing phase is defined again by the methodology outlined in `@mcmahon:2018` whereby the landing phase's end is defined as the point where the estimated center of mass velocity becomes greater than, or equal to, 0 meters per second (given that a negative velocity represents a downward movement).

The maximum landing force, average landing force, landing time, landing displacement, as well as various landing rate of force development metrics, are all computed during the landing phase of the jumps.

# Tables

Table 1. Metrics computed and exported by `JumpMetrics` at Takeoff for Countermovement Jumps (CMJ) and Squat Jumps (SQJ).

| Metric | CMJ | SQJ |
|--------|-----|-----|
| propulsive_peakforce_rfd_slope_between_events | ✓ | ✓ |
| propulsive_peakforce_rfd_instantaneous_average_between_events | ✓ | ✓ |
| propulsive_peakforce_rfd_instantaneous_peak_between_events | ✓ | ✓ |
| braking_peakforce_rfd_slope_between_events | ✓ | |
| braking_peakforce_rfd_instantaneous_average_between_events | ✓ | |
| braking_peakforce_rfd_instantaneous_peak_between_events | ✓ | |
| braking_propulsive_rfd_slope_between_events | ✓ | |
| braking_propulsive_rfd_instantaneous_average_between_events | ✓ | |
| braking_propulsive_rfd_instantaneous_peak_between_events | ✓ | |
| braking_net_vertical_impulse | ✓ | |
| propulsive_net_vertical_impulse | ✓ | |
| braking_to_propulsive_net_vertical_impulse | ✓ | |
| total_net_vertical_impulse | ✓ | ✓ |
| peak_force | ✓ | ✓ |
| maximum_force | ✓ | ✓ |
| average_force_of_braking_phase | ✓ | |
| average_force_of_propulsive_phase | ✓ | ✓ |
| takeoff_velocity | ✓ | ✓ |
| jump_height_takeoff_velocity | ✓ | ✓ |
| jump_height_net_vertical_impulse | ✓ | ✓ |
| movement_time | ✓ | ✓ |
| unweighting_time | ✓ | |
| braking_time | ✓ | |
| propulsive_time | ✓ | ✓ |
| lowering_displacement | ✓ | |
| frame_start_of_unweighting_phase | ✓ | |
| frame_start_of_breaking_phase | ✓ | |
| frame_start_of_propulsive_phase | ✓ | ✓ |
| frame_peak_force | ✓ | ✓ |
| frame_of_potential_unweighting_start | | ✓ |

# Acknowledgements

We acknowledge contributions from Malinda Hapuarachchi for providing the data required to develop, test, and verify this package.

# References