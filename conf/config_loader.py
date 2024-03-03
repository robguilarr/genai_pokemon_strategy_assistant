import yaml


def load_default_messages():
    with open("conf/default_messages.yml", "r") as file:
        return yaml.safe_load(file)


def load_global_conf():
    with open("conf/global_conf.yml", "r") as file:
        return yaml.safe_load(file)


def load_prompt_template_library():
    with open("conf/prompt_template_library.yml", "r") as file:
        return yaml.safe_load(file)


default_messages = load_default_messages()
global_conf = load_global_conf()
prompt_template_library = load_prompt_template_library()
