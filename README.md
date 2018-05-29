# joint-localization
Tinkerings with joint localization from optical marker motion capture system.

MarkerGroups.py: Segment markers into groups that are attached to rigid bodies through spectral clustering. Works well and fast for a small number of rigid bodies, but is yet too slow for full body marker sets.

MarkerTrajectoryBases.py: Estimate joint trajectories (Center of Rotation) connecting marker groups.
Does not yet compute the minimal spanning tree, so all pairs of marker groups
are still considered to be connected by a joint.
Due to the method this only works "offline" and not in a live session.

EstimateSphereDemo.py: Demonstrate the principle of estimating the CoR by
assuming marker positions are distributed on a spherical surface.
