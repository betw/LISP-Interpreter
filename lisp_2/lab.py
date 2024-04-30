"""
6.101 Lab 13:
LISP Interpreter Part 2
"""

#!/usr/bin/env python3
import sys

sys.setrecursionlimit(20_000)

# KEEP THE ABOVE LINES INTACT, BUT REPLACE THIS COMMENT WITH YOUR lab.py FROM
# THE PREVIOUS LAB, WHICH SHOULD BE THE STARTING POINT FOR THIS LAB.

"""
6.101 Lab 12:
LISP Interpreter Part 1
"""

#!/usr/bin/env python3

import sys
import doctest

sys.setrecursionlimit(20_000)

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def is_number(value):
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    tknd = source.replace("(", " ( ").replace(")", " ) ")

    while ";" in tknd:
        comment_index = tknd.find(";")
        newline_index = len(tknd)
        if "\n" in tknd:
            newline_index = tknd.find("\n")
        tknd = tknd[:comment_index] + tknd[newline_index + 1 :]

    tknd.replace("\n", " ")
    return tknd.split()


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """

    def parse_expression(index):
        token = number_or_symbol(tokens[index])
        # example: first is closing paren.
        if token == ")":
            raise SchemeSyntaxError(
                "Closing parentheses shouldn't appear early in expression."
            )
        if isinstance(token, (int, float)):
            return token, index + 1
        if token != "(":
            return token, index + 1
        else:
            output_expression = []
            index += 1
            # while there is things left in the expression
            # parse every sub_expression
            while index < len(tokens) and tokens[index] != ")":
                right, index = parse_expression(index)
                output_expression.append(right)

            # skip trailing parentheses
            return output_expression, index + 1

    parsed_exp, index = parse_expression(0)
    # too many closing parentheses
    if index != len(tokens):
        raise SchemeSyntaxError("There exist too many parentheses.")
    return parsed_exp


######################
# Built-in Functions #
######################


def mul(args):
    if len(args) == 1:
        return args[0]
    if len(args) == 0:
        return 1
    return args[0] * mul(args[1:])


def div(args):
    if len(args) == 0:
        raise SchemeEvaluationError("Division with no arguments.")
    if len(args) == 1:
        return 1 / args[0]
    return args[0] / mul(args[1:])


def equal(args):
    first_num = args[0]
    for arg in args:
        if arg != first_num:
            return False
    return True


def lessthan_greaterthan(args, type):
    for i in range(len(args) - 1):
        if type == ">":
            # decreasing order
            if args[i] <= args[i + 1]:
                return False
        elif type == ">=":
            # nonincreasing order
            if args[i] < args[i + 1]:
                return False
        elif type == "<":
            # increasing order
            if args[i] >= args[i + 1]:
                return False
        elif type == "<=":
            # nondecreasing order
            if args[i] > args[i + 1]:
                return False
    return True


def _and_(args, frame):
    for arg in args:
        if not evaluate(arg, frame):
            return False
    return True


def _or_(args, frame):
    for arg in args:
        if evaluate(arg, frame):
            return True
    return False


def _not_(args):
    if len(args) != 1:
        raise SchemeEvaluationError(
            "Only one argument allowed to NOT built-in function."
        )
    return not (args[0])


def cons(args):
    if len(args) != 2:
        raise SchemeEvaluationError(
            "Wrong number of arguments passed into CONS. Need two."
        )
    return Pair(args[0], args[1])


def car_or_cdr(args, type):
    if len(args) != 1:
        raise SchemeEvaluationError("Wrong number of arguments passed into CAR.")
    if not isinstance(args[0], Pair):
        raise SchemeEvaluationError("Argument is not of Pair type.")
    pair = args[0]
    if type == "car":
        return pair.get_car()
    if type == "cdr":
        return pair.get_cdr()


def _list_(args):
    if len(args) == 0:
        return None
    return Pair(args[0], _list_(args[1:]))


def is_list(args):
    linked_list = args[0]
    if linked_list is None:
        return True
    if isinstance(linked_list, Pair):
        if linked_list.get_cdr() == None or isinstance(linked_list.get_cdr(), Pair):
            return is_list([linked_list.get_cdr()])
        return False
    return False


def _length_(args):
    list_arg = args[0]
    if not is_list([list_arg]):
        raise SchemeEvaluationError("Argument is not of List type.")
    length = 0
    while list_arg != None:
        list_arg = list_arg.get_cdr()
        length += 1
    return length


def _listref_(args):
    list_arg = args[0]
    index = args[1]
    if index < 0:
        raise SchemeEvaluationError("Index must be nonegative.")
    elif not isinstance(list_arg, Pair):
        raise SchemeEvaluationError("Argument is not of Pair type.")
    elif not is_list([list_arg]):
        if index != 0:
            raise SchemeEvaluationError("Cons cell only supports index 0.")
        return list_arg.get_car()
    else:
        # list_arg is a list
        if index >= _length_([list_arg]):
            raise SchemeEvaluationError("Index OUT OF BOUNDS.")
        while index > 0:
            list_arg = list_arg.get_cdr()
            index -= 1
        return list_arg.get_car()


def _copy_(list_arg):
    """
    Return a shallow copy of the linked list, composed
    of Pair objects.
    """
    current = list_arg
    if current is None:
        return None
    # if current.get_cdr() is None:
    #     return Pair(current.get_car(), None)
    return Pair(current.get_car(), _copy_(current.get_cdr()))


def apply(function, linked_list):
    current = linked_list
    while current != None:
        current.set_car(function([current.get_car()]))
        current = current.get_cdr()
    return linked_list


def _map_(args):
    function = args[0]
    list_of_vals = args[1:][0]
    if not callable(function) or not is_list([list_of_vals]):
        raise SchemeEvaluationError(
            "Argument(s) not a callable function or not a list."
        )
    new_list = apply(function, _copy_(list_of_vals))
    return new_list


def _filter_(args):
    """
    Return a new list with all of the nodes for which the filter function
    returned false removed.
    """
    function = args[0]
    list_of_vals = args[1:]
    if not callable(function) or not is_list(list_of_vals):
        raise SchemeEvaluationError(
            "Argument(s) not a callable function or not a list."
        )
    new_list = _copy_(list_of_vals[0])
    current = new_list
    if current is None:
        return None

    # if the function returns False for the first value
    # in the linked list, return the rest of the list
    if _length_([current]) == 1 and not function([current.get_car()]):
        return None
    while current.get_cdr() != None:
        current_node = current.get_cdr()
        if not function([current_node.get_car()]):
            # remove the current node
            next_node = current_node.get_cdr()
            current.set_cdr(next_node)
            current_node.set_cdr(None)
        # if a node wasn't removed advance to the next node
        else:
            current = current.get_cdr()

    # check the first node
    if not function([new_list.get_car()]):
        temp = new_list.get_cdr()
        new_list.set_cdr(None)
        new_list = temp

    return new_list


def _reduce_(args):
    function = args[0]
    list_of_vals = args[1]
    initial_value = args[2]
    if not callable(function) or not is_list([list_of_vals]):
        raise SchemeEvaluationError(
            "Argument(s) not a callable function or not a list."
        )

    current = list_of_vals
    result_so_far = initial_value
    if current is None:
        return result_so_far
    while current != None:
        result_so_far = function([result_so_far, current.get_car()])
        current = current.get_cdr()
    return result_so_far


def combine_lists(new_list, next_list):
    pointer = new_list
    if new_list is None:
        return next_list
    while pointer.get_cdr() != None:
        pointer = pointer.get_cdr()
    pointer.set_cdr(next_list)
    return new_list


def _append_(args):
    if len(args) == 0:
        return None
    if not is_list([args[0]]):
        raise SchemeEvaluationError(
            "Argument to append only accepts objects of List type."
        )

    new_list = _copy_(args[0])
    # loop through each list and combine the list so far and next list
    for i in range(1, len(args)):
        if not is_list([args[i]]):
            raise SchemeEvaluationError(
                "Argument to append only accepts objects of List type."
            )
        next_list = _copy_(args[i])
        new_list = combine_lists(new_list, next_list)
    return new_list


scheme_builtins = {
    "#t": True,
    "#f": False,
    "equal?": equal,
    ">": lambda args: lessthan_greaterthan(args, ">"),
    ">=": lambda args: lessthan_greaterthan(args, ">="),
    "<": lambda args: lessthan_greaterthan(args, "<"),
    "<=": lambda args: lessthan_greaterthan(args, "<="),
    "and": _and_,
    "or": _or_,
    "not": _not_,
    "cons": cons,
    "list": _list_,
    "nil": None,
    "car": lambda args: car_or_cdr(args, "car"),
    "cdr": lambda args: car_or_cdr(args, "cdr"),
    "list?": is_list,
    "length": _length_,
    "list-ref": _listref_,
    "append": _append_,
    "map": _map_,
    "filter": _filter_,
    "reduce": _reduce_,
    "begin": lambda args: args[-1],
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": mul,
    "/": div,
}


##############
# Evaluation #
##############


def valid_variable(variable):
    """
    Return true if the variable represents a valid
    name, i.e. it doesn't contain spaces or parentheses
    and it is not a number.
    """
    if (
        not isinstance(variable, (int, float))
        and "(" not in variable
        and ")" not in variable
        and " " not in variable
    ):
        return True
    return False


def evaluate_file(filename, frame=None):
    with open(filename, "r") as f:
        expression = parse(tokenize("".join(f.read().split("\n"))))
        return evaluate(expression, frame)


class Frame:
    """
    A class representing a program frame in which
    it is stored the parent pointer and local variables.
    The frame can lookup variables, following parent frames
    and bind variables.
    """

    def __init__(self):
        self.parent = None
        self.local_variables = {}

    def lookup_var_frame(self, variable):
        if variable in self.local_variables:
            return self.local_variables
        elif self.parent != None:
            return self.parent.lookup_var_frame(variable)
        else:
            raise SchemeNameError(f"Variable with name {variable} is not defined.")

    def lookup_var(self, variable):
        return self.lookup_var_frame(variable)[variable]

    def bind_variable(self, variable, value):
        if valid_variable(variable):
            self.local_variables[variable] = value
            return value
        raise SchemeEvaluationError("Invalid name for variable.")


class Function:
    """
    A class representing a function, in which it is stored
    the enclosing frame pointer, the body of the function,
    and the parameters. The function behaviors include calling
    itself by creating a new frame and evaluating
    the code in its body.
    """

    def __init__(self, frame, parameters, body):
        self.enclosing_frame = frame
        self.body = body
        self.parameters = parameters

    def __call__(self, args):
        # make a new frame and bind it to
        # enclosing frame pointer
        new_frame = Frame()
        new_frame.parent = self.enclosing_frame

        # bind each parameter to their arguments
        if len(self.parameters) != len(args):
            raise SchemeEvaluationError("Incorrect number of arguments passed in.")
        for ix, parameter in enumerate(self.parameters):
            new_frame.bind_variable(parameter, args[ix])

        return evaluate(self.body, new_frame)


class Pair:
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    def get_car(self):
        return self.car

    def get_cdr(self):
        return self.cdr

    def set_cdr(self, cdr):
        self.cdr = cdr

    def set_car(self, car):
        self.car = car


builtins_frame = Frame()
builtins_frame.local_variables = scheme_builtins
global_frame = Frame()
global_frame.parent = builtins_frame


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if not frame:
        frame = global_frame
    if tree == []:
        raise SchemeEvaluationError("Function or operator is missing.")
    if isinstance(tree, str):
        return frame.lookup_var(tree)
    elif isinstance(tree, (int, float)):
        return tree
    elif tree[0] == "define":
        # if name itself is an expression, decompose into
        # a lambda
        if isinstance(tree[1], list):
            function_name, parameters = tree[1][0], tree[1][1:]
            tree[1] = function_name
            tree[2] = ["lambda", parameters, tree[2]]

        return frame.bind_variable(tree[1], evaluate(tree[2], frame))
    elif tree[0] == "lambda":
        return Function(frame, tree[1], tree[2])
    elif tree[0] == "if":
        predicate = evaluate(tree[1], frame)
        if predicate:
            return evaluate(tree[2], frame)
        return evaluate(tree[3], frame)
    elif tree[0] == "and" or tree[0] == "or":
        return evaluate(tree[0])(tree[1:], frame)
    elif tree[0] == "del":
        var_name = tree[1]
        try:
            value = frame.local_variables[var_name]
            del frame.local_variables[var_name]
            return value
        except KeyError:
            raise SchemeNameError(f"Local variable {var_name} was not defined.")
    elif tree[0] == "let":
        vars_and_vals, body = tree[1], tree[2]

        # bind the evaluated values to the variables in a new frame
        new_frame = Frame()
        new_frame.parent = frame
        for var_val in vars_and_vals:
            variable = var_val[0]
            value = evaluate(var_val[1], frame)
            new_frame.bind_variable(variable, value)

        # evaluate body in new frame
        return evaluate(body, new_frame)
    elif tree[0] == "set!":
        variable = tree[1]
        expression = evaluate(tree[2], frame)

        frame.lookup_var_frame(variable)[variable] = expression
        return expression
    elif isinstance(tree, list):
        # first element function
        if is_number(tree[0]):
            raise SchemeEvaluationError("Number is not an operator or function.")
        func = evaluate(tree[0], frame)
        if not callable(func):
            raise SchemeEvaluationError("Function is not callable!")
        args = [evaluate(elem, frame) for elem in tree[1:]]
        return func(args)


def result_and_frame(tree, frame=None):
    # frame is given, evaluate expression in specified frame
    if frame:
        return evaluate(tree, frame), frame
    # frame is not given, make brand new frame and evaluate there
    new_frame = Frame()
    new_frame.parent = global_frame
    return evaluate(tree, new_frame), new_frame


########
# REPL #
########

import os
import re
import sys
import traceback
from cmd import Cmd

try:
    import readline
except:
    readline = None


def supports_color():
    """
    Returns True if the running system's terminal supports color, and False
    otherwise.  Not guaranteed to work in all cases, but maybe in most?
    """
    plat = sys.platform
    supported_platform = plat != "Pocket PC" and (
        plat != "win32" or "ANSICON" in os.environ
    )
    # IDLE does not support colors
    if "idlelib" in sys.modules:
        return False
    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True


class SchemeREPL(Cmd):
    """
    Class that implements a Read-Evaluate-Print Loop for our Scheme
    interpreter.
    """

    history_file = os.path.join(os.path.expanduser("~"), ".6101_scheme_history")

    if supports_color():
        prompt = "\033[96min>\033[0m "
        value_msg = "  out> \033[92m\033[1m%r\033[0m"
        error_msg = "  \033[91mEXCEPTION!! %s\033[0m"
    else:
        prompt = "in> "
        value_msg = "  out> %r"
        error_msg = "  EXCEPTION!! %s"

    keywords = {
        "define",
        "lambda",
        "if",
        "equal?",
        "<",
        "<=",
        ">",
        ">=",
        "and",
        "or",
        "del",
        "let",
        "set!",
        "+",
        "-",
        "*",
        "/",
        "#t",
        "#f",
        "not",
        "nil",
        "cons",
        "list",
        "cat",
        "cdr",
        "list-ref",
        "length",
        "append",
        "begin",
    }

    def __init__(self, use_frames=False, verbose=False):
        self.verbose = verbose
        self.use_frames = use_frames
        self.global_frame = None
        Cmd.__init__(self)

    def preloop(self):
        if readline and os.path.isfile(self.history_file):
            readline.read_history_file(self.history_file)

    def postloop(self):
        if readline:
            readline.set_history_length(10_000)
            readline.write_history_file(self.history_file)

    def completedefault(self, text, line, begidx, endidx):
        try:
            bound_vars = set(self.global_frame)
        except:
            bound_vars = set()
        return sorted(i for i in (self.keywords | bound_vars) if i.startswith(text))

    def onecmd(self, line):
        if line in {"EOF", "quit", "QUIT"}:
            print()
            print("bye bye!")
            return True

        elif not line.strip():
            return False

        try:
            token_list = tokenize(line)
            if self.verbose:
                print("tokens>", token_list)
            expression = parse(token_list)
            if self.verbose:
                print("expression>", expression)
            if self.use_frames:
                output, self.global_frame = result_and_frame(
                    *(
                        (expression, self.global_frame)
                        if self.global_frame is not None
                        else (expression,)
                    )
                )
            else:
                output = evaluate(expression)
            print(self.value_msg % output)
        except SchemeError as e:
            if self.verbose:
                traceback.print_tb(e.__traceback__)
                print(self.error_msg.replace("%s", "%r") % e)
            else:
                print(self.error_msg % e)

        return False

    completenames = completedefault

    def cmdloop(self, intro=None):
        while True:
            try:
                Cmd.cmdloop(self, intro=None)
                break
            except KeyboardInterrupt:
                print("^C")


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)
    files = sys.argv[1:]
    for file in files:
        evaluate_file(file, global_frame)
    print("fib" in global_frame.local_variables)
    SchemeREPL(use_frames=True, verbose=True).cmdloop()
    # tknd1 = tokenize("\n(cat (dog (tomato)))")
    # tknd2 = tokenize(";add the numbers 2 and 3 \n (+ ;
    # this expression \n 2
    # ; spans multiple \n 3  ; lines \n )")
    # test_parse1 = parse(['(', '+', '2', '(', '-', '5', '3', ')', '7', '8', ')'])
    # test_parse2 = parse(['2'])
    # test_parse3 = parse(['x'])
    # test_parse4 = parse(tokenize('(define circle-area (lambda (r) (* 3.14 (* r r))))'))
    # print(parse(['(', ')']))
    # print(test_parse1)
    # print(test_parse2)
    # print(test_parse3)
    # print(test_parse4)

    # testing syntax errors for parse
    # parentheses
    # test_parse1 = parse(["(", "("])
    # print(test_parse1)

    # testing evaluate
    # print(evaluate(['+', 3, 7, 2]))
    # print(evaluate(['+', 3, ['-', 7, 5]]))
    # print(evaluate(3.14))
    # print(evaluate('+'))

    # print(parse(tknd1))
    # print(div([2, 3, 4]))
    # print(div([9, 7]))
    pass
