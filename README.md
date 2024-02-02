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
***
# Output

Learning process:

![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/351b3713-4040-40ce-bfaa-f50379b5085f)

Observed vs simulated for all depths:

![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/134719cf-074e-456f-99ed-313a80a9d542)
![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/56765479-c272-4b98-8bde-4461251f6e38)
![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/4acc1575-d5fa-4ae0-9ac0-1998972a4b92)
![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/8cb83d46-f71b-46c0-a2ba-151295eb8d52)
![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/3023f128-db77-4dd1-a49a-aa1cfed77245)
![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/cd1eafab-ed59-419f-8904-264a1857baba)
![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/ee9d62f5-6d4e-43c3-b438-f08470d9baf1)

Predicted advective flow velocities:
![data](https://github.com/saluisto/PINN_HEAT/assets/151910262/c5001fd6-7894-452c-8941-01ccce3e4a2a)
