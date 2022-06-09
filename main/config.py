import string
from configparser import ConfigParser
import itertools
import warnings

from requests import get

warnings.filterwarnings('default')
config_file = 'config.ini'


def get_property(section:str, property:str)->str:
    config = ConfigParser()
    config.read(config_file)
    try:
        return config[section][property]
    except: 
        print(f'property not found : {section,property}')
        return ''

if __name__=='__main__':
    p = get_property('DATA','train')
    print(f'path : {p}')
