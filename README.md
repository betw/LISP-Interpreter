Credit: 6.1900 Staff for template

This repository contains a custom Scheme interpreter implemented in Python, supporting a variety of fundamental Scheme functionalities, including symbolic computation, list processing, and mathematical operations. The interpreter provides a robust environment for evaluating Scheme expressions, complete with built-in functions like +, *, car, cdr, map, and custom functions via lambda. It is designed to handle nested expressions, user-defined variables, conditionals (if), and higher-order functions like filter and reduce.

Example of Functionality
You can interact with the interpreter through a Read-Eval-Print Loop (REPL). For example, defining and using a function to calculate the factorial of a number is as simple as:

```
(define (factorial n)
    (if (= n 0)
        1
        (* n (factorial (- n 1)))))
```
Then, calling ```(factorial 5)``` will correctly evaluate to 120. The interpreter also supports list processing, allowing operations like:

```
(map (lambda (x) (* x x)) (list 1 2 3 4));
Returns a list (1 4 9 16)
```
This project is a valuable tool for understanding the principles of Scheme programming and functional paradigms, while also providing extensible functionality to explore custom implementations of Scheme features.
