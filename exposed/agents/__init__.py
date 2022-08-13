import inspect
import importlib

def get_agent(name, env, exp):

    module_name, *attribs = name.split(':')

    try:
        module = importlib.import_module('.' + module_name, package=__package__)
    except ModuleNotFoundError:
        module = importlib.import_module(module_name, package=__package__)

    loader = module
    for attrib in attribs:
        loader = getattr(loader, attrib)
    if inspect.ismodule(loader):
        loader = loader.load

    return loader(env, exp)
