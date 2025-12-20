from wildtrain.trainers.sweeps import ClassifierSweeper



def main():

    sweeper = ClassifierSweeper(r"D:\workspace\repos\wildtrain\configs\classification\classification_sweep.yaml",debug=True)
    sweeper.run()


if __name__ == "__main__":
    main()