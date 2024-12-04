# main.py

from PDE_model import ForestFirePDE

def main():
    # Parameters
    k = 0.1
    epsilon = 0.3
    alpha = 0.001
    u_pc = 3
    q = 1
    spatial_length = 90
    spatial_resolution = 128
    temporal_length = 1000 # total time = 1000 * 25 / 500 = 50
    delta_t = 25 / 500

    # Initialize and run the model
    model = ForestFirePDE(k, epsilon, alpha, u_pc, q, spatial_length, spatial_resolution, temporal_length, delta_t)
    model.run()

    # Plot at different times
    model.plot_t_and_f(0,False) # plot at time 0 and savefig = False
    model.plot_t_and_f(30, False) # plot at time 30 and savefig = False

if __name__ == "__main__":
    main()