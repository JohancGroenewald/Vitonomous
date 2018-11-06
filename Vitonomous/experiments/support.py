
def display(*args):
    import inspect
    calling_frame = inspect.currentframe().f_back
    calling_context = inspect.getframeinfo(calling_frame, 1).code_context[0]
    arguments = calling_context.split('(', 1)[1].split(')')[0]
    s = ['-> ']
    for i, argument in enumerate(arguments.split(',')):
        s.append('{:.<4}{:.>4}'.format(argument, str(args[i])))
    print(' |'.join(s))
