from simulation_pkg import basic

def main():
    basic.reset_model("ego_vehicle", "prius_hybrid", basic.driving_ego())
    
if __name__ == "__main__":
    main()
