from run.config import Configurable

config = Configurable('/home/clementine/projects/Dynet-Biaffine-dependency-parser/configs/default.cfg', extra_args='--cccc special')
print(config.__dict__)
