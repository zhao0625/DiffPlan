from sacred import Experiment

from sacred import SETTINGS
# > for some capturing error
SETTINGS['CAPTURE_MODE'] = 'sys'
# > remove read-only class in config
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# > start experiment
ex = Experiment('SymPlan-Main')
