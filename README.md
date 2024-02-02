# PINN_HEAT
Python code for the inverse solution of the heat covection/conduction equation using physics informed neural networks.

## Aim of the project
The code is designed to perform inverse estimation of groundwater fluxes by leveraging measured temperature profiles in the subsurface. This is achieved through the inverse solution of the heat conduction/convection equation . The inverse simulation employs a Physics-Informed Neural Network (PINN) approach, implemented using Python, TensorFlow, and the deepxde library.

## Why is the project useful?
Traditionally, inverse simulations rely on numerical or analytical solutions of the heat transport equation. However, these methods exhibit insensitivity to high-frequency flux variations, often encountered in dynamic environments such as high-energy beach faces or tidal systems. The presented proof-of-concept PINN framework overcomes this limitation, demonstrating a notable capability to accurately quantify groundwater fluxes in such systems. Its resilience to high-frequency variations suggests a promising potential for reliable applications in these dynamic environments.

## How can the community contribute?
The PINN network is currently a work in progress and has only been applied to a relatively limited dataset. The learning process of the PINN is not yet optimized, and convergence remains slow. The primary objective for the future is to enhance the efficiency of the PINN, enabling its application to extended and finely resolved datasets. Community support and collaboration are highly appreciated in achieving this goal. Contributions and insights from the community would greatly aid in refining the PINN's performance for broader and more complex datasets.

##Where can the community can get assitance?
sven.frei@wur.nl
***
# Data
The PINN_heat uses a synthetic dataset of observed temperatures for a 7 different depths for the inverse solution of the advective flow velocities:  

![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/d92e90fa-21cd-42da-9a36-ddd7b456c22b)

The synthetic timeseries was generated using a numerical solution to the heat transport equation using predefined advective flow velocities:

![Flow](https://github.com/saluisto/PINN_HEAT/assets/151910262/dd7b9a07-5bde-492c-9786-76a2c4da1696)

The code for generating the data is publicly avaiable under https://doi.org/10.5066/P99DBTKT

# Output
