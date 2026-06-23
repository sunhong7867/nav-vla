from simulation_pkg import basic

def main():
    for entity_name, model_name, pose in basic.mission_obstacle_specs():
        basic.load_model(entity_name, model_name, pose)

if __name__ == "__main__":
    main()
