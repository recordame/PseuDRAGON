"""Exact archived bounded-expression validator; no product dependency."""
import ast


def checked_expression(code):
    """Permit only small lambda expressions with explicit safe capabilities."""
    if not isinstance(code, str) or not code.strip() or len(code) > 4096:
        raise ValueError('Expected a nonempty lambda of at most 4096 characters')
    tree = ast.parse(code, mode='eval')
    if not isinstance(tree.body, ast.Lambda):
        raise ValueError('Only a lambda expression can run in the raw evaluator')
    allowed_nodes = (ast.Expression, ast.Lambda, ast.arguments, ast.arg, ast.Name,
        ast.Load, ast.Store, ast.Constant, ast.IfExp, ast.BoolOp, ast.And, ast.Or,
        ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub,
        ast.Mult, ast.Mod, ast.FloorDiv, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
        ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
        ast.Subscript, ast.Slice, ast.List, ast.Tuple, ast.Call, ast.Attribute,
        ast.keyword, ast.JoinedStr, ast.FormattedValue, ast.ListComp,
        ast.GeneratorExp, ast.comprehension)
    attributes = {'notna', 'isna', 'sub', 'findall', 'match', 'fullmatch', 'group',
        'isdigit', 'isdecimal', 'isnumeric', 'replace', 'strip', 'lstrip', 'rstrip',
        'upper', 'lower', 'join', 'startswith', 'endswith'}
    builtins = {'str', 'len', 'min', 'max', 'enumerate', 'range', 'list', 'tuple',
                'reversed', 'isinstance', 'int', 'pd', 're'}
    variables = {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    variables |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    if any(name.startswith('_') for name in variables):
        raise ValueError('Private variable names are not supported')
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError('Unsupported AST node: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in variables | builtins:
            raise ValueError('Unsupported name: ' + node.id)
        if isinstance(node, ast.Attribute) and node.attr not in attributes:
            raise ValueError('Unsupported attribute: ' + node.attr)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (str, int, type(None), bool)):
                raise ValueError('Unsupported constant type')
            if isinstance(node.value, str) and len(node.value) > 256:
                raise ValueError('Oversized string constant')
            if isinstance(node.value, int) and abs(node.value) > 1000:
                raise ValueError('Oversized integer constant')
        if isinstance(node, ast.keyword) and node.arg is None:
            raise ValueError('Expanded keyword arguments are not supported')
        if isinstance(node, ast.comprehension) and node.is_async:
            raise ValueError('Asynchronous expressions are not supported')
    return tree
