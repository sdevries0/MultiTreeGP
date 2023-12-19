import jax.numpy as jnp
import sympy

from networkTrees import NetworkTrees

def tree_to_string(tree: list):
    "Transform tree to string"
    if len(tree)==3:
        left_tree = "(" + tree_to_string(tree[1]) + ")" if len(tree[1]) == 3 else tree_to_string(tree[1])
        right_tree = "(" + tree_to_string(tree[2]) + ")" if len(tree[2]) == 3 else tree_to_string(tree[2])
        return left_tree + tree[0] + right_tree
    elif len(tree)==2:
        return tree[0] + "(" + tree_to_string(tree[1]) + ")"
    else:
        return str(tree[0])
    
#Simplification methods
def tree_to_sympy(tree: list, eval: bool = True):
    "Transforms a tree to sympy format"
    str_sol = tree_to_string(tree)
    expr = sympy.parsing.sympy_parser.parse_expr(str_sol, evaluate=eval)
    
    return expr

def trees_to_sympy(trees: NetworkTrees):
    "Transforms trees to sympy formats"
    sympy_trees = []
    for tree in trees():
        sympy_trees.append(tree_to_sympy(tree))
    return sympy_trees

def replace_negatives(tree: list):
    "Replaces '+-1.0*x' with '-x' to simplify trees even further"
    if len(tree)<3:
        return tree
    left_tree = replace_negatives(tree[1])
    right_tree = replace_negatives(tree[2])

    if tree[0]=="+" and right_tree[0]=="*" and right_tree[1]==-1.0:
        return ["-", left_tree, right_tree[2]]
    return [tree[0], left_tree, right_tree]

def sympy_to_tree(sympy_expr, mode: str):
    "Reconstruct a tree from a sympy expression"
    if isinstance(sympy_expr,sympy.Float) or isinstance(sympy_expr,sympy.Integer):
        return [jnp.array(float(sympy_expr))]
    elif isinstance(sympy_expr,sympy.Symbol):
        return [str(sympy_expr)]
    elif isinstance(sympy_expr,sympy.core.numbers.NegativeOne):
        return [jnp.array(-1)]
    elif isinstance(sympy_expr,sympy.core.numbers.Zero):
        return [jnp.array(0)]
    elif isinstance(sympy_expr,sympy.core.numbers.Half):
        return [jnp.array(0.5)]
    elif isinstance(sympy_expr*-1, sympy.core.numbers.Rational):
        return [jnp.array(float(sympy_expr))]
    elif not isinstance(sympy_expr,tuple):
        if isinstance(sympy_expr,sympy.Add):
            left_tree = sympy_to_tree(sympy_expr.args[0], "Add")
            if len(sympy_expr.args)>2:
                right_tree = sympy_to_tree(sympy_expr.args[1:], "Add")
            else:
                right_tree = sympy_to_tree(sympy_expr.args[1], "Add")
            return ["+",left_tree,right_tree]
        if isinstance(sympy_expr,sympy.Mul):
            left_tree = sympy_to_tree(sympy_expr.args[0], "Mul")
            if len(sympy_expr.args)>2:
                right_tree = sympy_to_tree(sympy_expr.args[1:], "Mul")
            else:
                right_tree = sympy_to_tree(sympy_expr.args[1], "Mul")
            return ["*",left_tree,right_tree]
        if isinstance(sympy_expr,sympy.cos):
            return ["cos",sympy_to_tree(sympy_expr.args[0], mode=None)]
        if isinstance(sympy_expr,sympy.sin):
            return ["sin",sympy_to_tree(sympy_expr.args[0], mode=None)]
        if isinstance(sympy_expr, sympy.Pow):
            if sympy_expr.args[1]==-1:
                right_tree = sympy_to_tree(sympy_expr.args[0], "Mul")
                return ["/",[jnp.array(1)],right_tree]
            else:
                left_tree = sympy_to_tree(sympy_expr.args[0], "Add")
                right_tree = sympy_to_tree(sympy_expr.args[1], "Add")
                return ["**", left_tree, right_tree]
    else:
        if mode=="Add":
            left_tree = sympy_to_tree(sympy_expr[0], "Add")
            if len(sympy_expr)>2:
                right_tree = sympy_to_tree(sympy_expr[1:], "Add")
            else:
                right_tree = sympy_to_tree(sympy_expr[1], "Add")
            return ["+",left_tree,right_tree]
        if mode=="Mul":
            left_tree = sympy_to_tree(sympy_expr[0], "Mul")
            if len(sympy_expr)>2:
                right_tree = sympy_to_tree(sympy_expr[1:], "Mul")
            else:
                right_tree = sympy_to_tree(sympy_expr[1], "Mul")
            return ["*",left_tree,right_tree]
    
def simplify_tree(tree: list):
    "Simplifies a tree by transforming the tree to sympy format and reconstructing it"
    old_tree_string = tree_to_sympy(tree, eval=False)
    new_tree_string = tree_to_sympy(tree, eval=True)

    if old_tree_string==new_tree_string:
        return False

    if new_tree_string==sympy.nan or new_tree_string.has(sympy.core.numbers.ImaginaryUnit, sympy.core.numbers.ComplexInfinity): #Return None when the sympy expression contains illegal terms
        return False

    new_tree = sympy_to_tree(new_tree_string, "Add" * isinstance(new_tree_string, sympy.Add) + "Mul" * isinstance(new_tree_string, sympy.Mul))
    new_tree = replace_negatives(new_tree)
    return new_tree
