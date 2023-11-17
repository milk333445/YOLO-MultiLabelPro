import yaml
from easydict import EasyDict



class ConfigManager:
       def __init__(self, filepath='./settings/autolabel_settings.yaml'):
              with open(filepath, 'r') as yaml_file:
                     self.data = yaml.safe_load(yaml_file)
              if 'classes' in self.data:
                  # Convert to String
                     self.data['classes'] = [str(x) for x in self.data['classes']]

       def get_config(self):
            conf = {} # EasyDict() doesn't handle key_actions.
            conf['obj'] = self.data['classes']
            conf['clr'] = [tuple(color) for color in self.data['settings']['clr']]
            conf['key_actions_normal'] = {int(k): v for k, v in self.data['key_actions_normal'].items()} # Convert string keys to integers
            conf['key_actions_obb'] = {int(k): v for k, v in self.data['key_actions_obb'].items()}
            return conf






