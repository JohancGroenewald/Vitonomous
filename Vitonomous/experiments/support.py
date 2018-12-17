
def display(*args):
    import inspect
    calling_frame = inspect.currentframe().f_back
    calling_context = inspect.getframeinfo(calling_frame, 1).code_context[0]
    arguments = calling_context.split('(', 1)[1].split(')')[0]
    s = ['-> ']
    bumper = False
    for i, argument in enumerate(arguments.split(',')):
        if argument == 'True':
            bumper = True
        else:
            s.append('{:_<4} {:_>4}'.format(argument, str(args[i])))
    buffer = ' |'.join(s)
    print(buffer)
    if bumper:
        print(len(buffer)*'-')
