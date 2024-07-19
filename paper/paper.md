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

Researchers and practitioners commonly use countermovement jumps and squat jumps to evaluate people's capacity `[@mcmahon:2017]`, determine potential risk for injury `[@bird:2022]`, or assess "readiness" for training `[@watkins:2017]`. Commonly, these jumps are performed on force plates to collect the relevant time series data to derive these insights. After collecting raw force data, `JumpMetrics` detects various events and computes metrics to derive the multiple insights the exercise professional wishes to glean from the individual. However, despite the necessity to use algorithms to consistently detect events and compute metrics from this time series data, there are currently no free, open-source alternatives for researchers and practitioners to leverage to ensure the data they are computing are consistent between groups.

# Statement of need

`JumpMetrics` is a free, open-source Python package for computing countermovement jump and squat jump events and metrics. The API for `JumpMetrics` was designed to be modular and easy to use. Researchers and practitioners can examine each jump's takeoff and landing phases individually or together, depending on their needs and research questions. Furthermore, `JumpMetrics` provides various helper functions to prepare the data for detecting events and computing metrics. These involve cropping and low-pass filtering functions if both steps are appropriate for a researcher or practitioner's particular analysis.

`JumpMetrics` was designed as a free and easy-to-use tool for researchers and practitioners to process single jumps or batch process several jumps consistently and openly. Beyond making analyses with force plates more accessible, `JumpMetrics` aims to ensure that the events and metrics are easy to compute and reproducible. `JumpMetrics` can also be leveraged by undergraduate or graduate students to assist them with sports science-related projects.

# How the Library Works
## Event Detections and Metrics
`JumpMetrics` computes various events and metrics for the takeoff and landing phases. The events (and thus metrics) are slightly different between the jump variations, given the differences in their movement executions.
### Takeoff Phase
#### Countermovement Jumps
The takeoff phase includes both the "lowering" (sometimes referred to as the eccentric) and "ascending" (sometimes referred to as the concentric) phases of the jump. The events computed in this package correspond to the phases of the countermovement jump highlighted in `@mcmahon:2018`. These involve the start of the "unweighting" phase, the start of the braking phase, the start of the propulsive phase, and the frame corresponding to the peak force. `JumpMetrics` computes the start of the unweighting phase in the same manner as outlined in `@owen:2014` whereby the first frame of force data that exceeds five times the standard deviation of the force data (default value; this is a tuneable parameter depending on the data collection parameters) during quiet standing defines the start of the unweighting phase. The braking phase starts at the frame corresponding to the minimum downward movement velocity (of the individual's estimated center of mass). The propulsive phase starts at the frame corresponding to the minimum downward displacement (of the individual's estimated center of mass). The peak force event is captured using `find_peaks` from the `scipy` package and looks for a "peak" in the force series.

![Example countermovement jump force-time trace with events detected during the takeoff phase.](/analyses/study_1/figures/F02/CTRL1/literature_cutoff/force.png){ width=25% }
Figure 1. Example countermovement jump force-time trace with events detected during the takeoff phase.


`JumpMetrics` computes the rate of force development, net vertical impulse, and average force between all events detected during the countermovement jump. Additionally, metrics such as the jump height (based on the net vertical impulse and using the impulse-momentum relationship as well as the velocity at the final frame of data before takeoff), takeoff velocity, movement time, unweighting time, braking time, propulsive time, lowering displacement are also all computed. Table 1 contains a complete list of the metrics `JumpMetrics` exports for countermovement jumps.

#### Squat Jumps
Given that the squat jump is intentionally performed with a pause to remove the continuous countermovement, the only events computed are the start of the propulsive phase and the peak force event. The start of the propulsive phase is the first frame of data that exceeds five times the standard deviation of the force data (default value; this is a tuneable parameter depending on the data collection parameters) during the squat phase. The peak force event is computed similarly to the countermovement jumps whereby `find_peaks` from the `scipy` package and looks for a "peak" in the force series.
![Example squat jump force-time trace during the takeoff phase.](/analyses/study_3/figures/SQT/P02/3_5/literature_cutoff/force.png){ width=25% }
Figure 2. Example squat jump force-time trace during the takeoff phase.

Although there is not supposed to be any countermovement during a squat jump, depending on the instructions and guidance provided to the participant, as well as their general movement behaviours, there may be a minor countermovement that would negate the trial from being a proper squat jump. `JumpMetrics` explicitly looks for and flags this motion (with a warning and an estimated frame in the metrics output) to make the user aware of this potential flaw in the squat jump trial.
![Example squat jump force-time trace with an inappropriate countermovement detected during the takeoff phase.](/analyses/study_3/figures/SQT/P02/1_2/literature_cutoff/force.png){ width=25% }
Figure 3. Example squat jump force-time trace with an inappropriate countermovement detected during the takeoff phase.

`JumpMetrics` computes the rate of force development, net vertical impulse, and average force between all events detected during the countermovement jump. Additionally, metrics such as the jump height (based on the net vertical impulse and using the impulse-momentum relationship and the velocity at the final frame of data before takeoff), takeoff velocity, movement time, and propulsive time are also all computed. Table 1 contains a complete list of the metrics `JumpMetrics` exports for squat jumps.

### Landing Phase
The landing phase is defined again by the methodology outlined in `@mcmahon:2018` whereby the landing phase's end is the point where the estimated center of mass velocity becomes greater than, or equal to, 0 meters per second (given that a negative velocity represents a downward movement).

During the jumps' landing phase, `JumpMetrics` can compute the maximum landing force, average landing force, landing time, landing displacement, and various landing rate of force development metrics.

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

We acknowledge contributions from Malinda Hapuarachchi for providing the data required to develop, test, and verify the functions in this package.

# References