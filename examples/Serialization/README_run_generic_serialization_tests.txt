This script is intended to demonstrate a simple way to validate a workflow with serialized populations can meet your needs.

The config.json is based on EMOD/Regression/Scenarios/Generic/03_SEIR with slightly lower infetivity, longer incubation, and longer infectious periods to get a longer epidemic.

Run the script thusly

> python run_generic_serialization_tests.py

The script creates an experiment with simulations in each of the following roles:
1) full run sim: This simulation simply runs the simulation from the Start_Time through the Simulation_Duration and completes.
2) serializing sim: These sims run to the serialization timestep specified and write out a statefile.
3) reloading sim: These sims load the statefile from the output directories of the serializing sims and continue the simulation to the final timestep

After the simulations have completed, the InsetChart.json files from the serializaing and reloading sims are stitched together into a new file called combined_InsetChart.json and written to the output folder of the reloading sim.

A copy of the InsetChart.json file for the fullrun sim is made, also with the name combined_InsetChart.json

A Komolgorov / Smirnov test is run against a specified list of InsetChart channels from the combined_InsetCharts, and tags are added to the reloading sims to indicate the PValues for the channels under test. The StatisticalPopulation channel is included as an example of a channel where the data is guaranteed to be precisely the same, to show what complete agreement looks like (PVal=1.0)

To view the outputs as a graph, view the simulation in COMPS like so:

https://comps.idmod.org/#explore/Simulations?filters=ExperimentId=2946ac14-926c-e911-a2c0-c4346bcb1554

(Note that this is viewing all of the simulations in the experiment, and that your experiment ID will be different)

Select all of the reloading sims and the fullrun sim with the checkboxes on the left, and select the "Chart" view.  This should open up the mutichart UI.

While in the "Chart" UI, click on the "Source" dropdown and select "combined_InsetChart.json" from the list of options. At this point you can select any channel from the Y Value list, and select "Serialization Timestep" from the "hover tag" list, and this should allow you to compare the simulation results. 
 