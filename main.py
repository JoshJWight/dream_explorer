import dream_explorer.modules.atari.atari_module as atari_module
import dream_explorer.modules.doom.doom_module as doom_module
import dream_explorer.modules.mario.mario_module as mario_module
import dream_explorer.dream_explorer as dream_explorer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--logdir', type=str)
    args = parser.parse_args()

    if args.task is None or args.logdir is None:
        print("Usage: main.py --task <taskname> --logdir <logdir>")
        exit(1)

    my_module = None
    if args.task == "mario":
        my_module = mario_module.MarioModule()
    elif args.task == "doom":
        my_module = doom_module.DoomModule()
    elif args.task == "atari":
        my_module = atari_module.AtariModule()
    else:
        print("Unknown task: " + args.task)
        exit(1)

    dream_explorer.run(my_module, args.logdir)