# > Import ex initialization from a separate script
from scripts.init import ex

# > Import modules to register components (used by decorators, don't remove)
from scripts import commands, helpers


@ex.main
def main(_log):
    # > Not using @ex.automain for easier importing ex object
    _log.warning('[Please use the commands in commands.py. No default main command implemented.]')


if __name__ == '__main__':
    ex.run_commandline()
