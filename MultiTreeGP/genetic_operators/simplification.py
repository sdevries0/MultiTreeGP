import jax.numpy as jnp
import sympy

from MultiTreeGP.expression import OperatorNode, LeafNode

def tree_to_string(tree: list) -> str:
    """Transform tree to string representation.

    :param tree: Tree of operators and leaves.
    :returns: String representation of tree.
    """
    if len(tree)==3:
        left_tree = "(" + tree_to_string(tree[1]) + ")" if len(tree[1]) > 1 else tree_to_string(tree[1])
        right_tree = "(" + tree_to_string(tree[2]) + ")" if len(tree[2]) > 1 else tree_to_string(tree[2])
        return left_tree + str(tree[0]) + right_tree
    elif len(tree)==2:
        if tree[0].in_front:
            return str(tree[0]) + "(" + tree_to_string(tree[1]) + ")"
        else:
            return "(" + tree_to_string(tree[1]) + ")" + str(tree[0])
    else:
        return str(tree[0])
    
#Simplification methods
def tree_to_sympy(tree: list, eval: bool = True) -> sympy.core.Expr:
    """Transforms a tree to sympy format.

    :param tree: Tree of operators and leaves.
    :param eval: Indicates if the trees should be simplified.
    :returns: Sympy representation of tree.
    """
    str_sol = tree_to_string(tree)
    expr = sympy.parsing.sympy_parser.parse_expr(str_sol, evaluate=eval)
    
    return expr

def replace_negatives(tree: list) -> list:
    """Replaces '+-1.0*x' with '-x' to simplify trees even further.

    :param tree: Tree of operators and leaves.
    :returns: Simplified tree.
    """
    if len(tree)<3:
        return tree
    left_tree = replace_negatives(tree[1])
    right_tree = replace_negatives(tree[2])

    if str(tree[0])=="+" and str(right_tree[0])=="*" and right_tree[1]==-1.0:
        return [OperatorNode(lambda x, y: x - y, "-"), left_tree, right_tree[2]]
    return [tree[0], left_tree, right_tree]

def sympy_to_tree(sympy_expr: sympy.core.Expr, mode: str) -> list:
    """Reconstruct a tree from a sympy expression.

    :param sympy_expr: Sympy representation of a tree.
    :param mode: Indicates if current subtree should be multiplied or added
    :returns: Tree.
    """
    if isinstance(sympy_expr,sympy.Float) or isinstance(sympy_expr,sympy.Integer):
        return [jnp.array(float(sympy_expr))]
    elif isinstance(sympy_expr,sympy.Symbol):
        symbol, index = str(sympy_expr)[:-1], str(sympy_expr)[-1]
        return [LeafNode(symbol, int(index))]
    elif isinstance(sympy_expr,sympy.core.numbers.NegativeOne):
        return [jnp.array(-1.0)]
    elif isinstance(sympy_expr,sympy.core.numbers.Zero):
        return [jnp.array(0.0)]
    elif isinstance(sympy_expr,sympy.core.numbers.Half):
        return [jnp.array(0.5)]
    elif isinstance(sympy_expr,sympy.core.numbers.Exp1):
        return [jnp.exp(1.0)]
    elif isinstance(sympy_expr*-1, sympy.core.numbers.Rational):
        return [jnp.array(float(sympy_expr))]
    elif not isinstance(sympy_expr,tuple):
        if isinstance(sympy_expr,sympy.Add):
            left_tree = sympy_to_tree(sympy_expr.args[0], "Add")
            if len(sympy_expr.args)>2:
                right_tree = sympy_to_tree(sympy_expr.args[1:], "Add")
            else:
                right_tree = sympy_to_tree(sympy_expr.args[1], "Add")
            return [OperatorNode(lambda x, y: x + y, "+", 2),left_tree,right_tree]
        if isinstance(sympy_expr,sympy.Mul):
            left_tree = sympy_to_tree(sympy_expr.args[0], "Mul")
            if len(sympy_expr.args)>2:
                right_tree = sympy_to_tree(sympy_expr.args[1:], "Mul")
            else:
                right_tree = sympy_to_tree(sympy_expr.args[1], "Mul")
            return [OperatorNode(lambda x, y: x * y, "*", 2),left_tree,right_tree]
        if isinstance(sympy_expr,sympy.cos):
            return [OperatorNode(lambda x: jnp.cos(x), "cos", 1),sympy_to_tree(sympy_expr.args[0], mode=None)]
        if isinstance(sympy_expr,sympy.sin):
            return [OperatorNode(lambda x: jnp.sin(x), "sin", 1),sympy_to_tree(sympy_expr.args[0], mode=None)]
        if isinstance(sympy_expr,sympy.tanh):
            return [OperatorNode(lambda x: jnp.tanh(x), "tanh", 1),sympy_to_tree(sympy_expr.args[0], mode=None)]
        if isinstance(sympy_expr,sympy.log):
            return [OperatorNode(lambda x: jnp.log(x), "log", 1),sympy_to_tree(sympy_expr.args[0], mode=None)]
        if isinstance(sympy_expr,sympy.exp):
            return [OperatorNode(lambda x: jnp.exp(x), "exp", 1),sympy_to_tree(sympy_expr.args[0], mode=None)]
        if isinstance(sympy_expr, sympy.Pow):
            if sympy_expr.args[1]==-1:
                right_tree = sympy_to_tree(sympy_expr.args[0], "Mul")
                return [OperatorNode(lambda x, y: x / y, "/", 2),[jnp.array(1.0)],right_tree]
            if sympy_expr.args[1]==2:
                right_tree = sympy_to_tree(sympy_expr.args[0], "Add")
                return [OperatorNode(lambda x: x ** 2, "**2", 1, in_front=False),right_tree]
            else:
                left_tree = sympy_to_tree(sympy_expr.args[0], "Add")
                right_tree = sympy_to_tree(sympy_expr.args[1], "Add")
                return [OperatorNode(lambda x, y: x ** y, "**", 2), left_tree, right_tree]
    else:
        if mode=="Add":
            left_tree = sympy_to_tree(sympy_expr[0], "Add")
            if len(sympy_expr)>2:
                right_tree = sympy_to_tree(sympy_expr[1:], "Add")
            else:
                right_tree = sympy_to_tree(sympy_expr[1], "Add")
            return [OperatorNode(lambda x, y: x + y, "+", 2),left_tree,right_tree]
        if mode=="Mul":
            left_tree = sympy_to_tree(sympy_expr[0], "Mul")
            if len(sympy_expr)>2:
                right_tree = sympy_to_tree(sympy_expr[1:], "Mul")
            else:
                right_tree = sympy_to_tree(sympy_expr[1], "Mul")
            return [OperatorNode(lambda x, y: x * y, "*", 2),left_tree,right_tree]
    
def simplify_tree(tree: list) -> list:
    """Simplifies a tree by transforming the tree to sympy format and reconstructing it.

    :param tree: Tree of operators and leaves.
    :returns: Simplified tree.
    """
    old_tree_string = tree_to_sympy(tree, eval=False)
    new_tree_string = tree_to_sympy(tree, eval=True)

    #Check if simplification was possible
    if old_tree_string==new_tree_string:
        return False

    #Check if the simplified expression contains illegal terms
    if new_tree_string==sympy.nan or new_tree_string.has(sympy.core.numbers.ImaginaryUnit, sympy.core.numbers.ComplexInfinity):
        return False

    #Reconstruct sympy expression to tree format
    new_tree = sympy_to_tree(new_tree_string, "Add" * isinstance(new_tree_string, sympy.Add) + "Mul" * isinstance(new_tree_string, sympy.Mul))
    new_tree = replace_negatives(new_tree)
    return new_tree
