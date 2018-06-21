from core.config import Config
from core.run import run

if __name__ == '__main__':
    print("Load new data with smaller map and start training from beginning with random initialization (if output folder is empty).")
    run(Config(root_folder="./resources/first_step", output_folder="/.././output"))

    print("Load new data with larger map and increase map and cathegories in second step (with random values for new ones and given values for old ones).")
    run(Config(root_folder="./resources/second_step", output_folder="/.././output"))
    input("Press Enter to continue...")
