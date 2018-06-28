Module(body=[
    FunctionDef(
        name='add',
        args=arguments(
            args=[
                arg(arg='arg1', annotation=None),
                arg(arg='arg2', annotation=None),
                arg(arg='arg3', annotation=None)
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]),
            body=[
                Return(
                    value=BinOp(left=BinOp(left=Name(id='arg1', ctx=Load()), op=Add(), right=Name(id='arg2', ctx=Load())), op=Add(), right=Name(id='arg3', ctx=Load()))
                )
            ],
            decorator_list=[],
            returns=None
    )
]
)


Module(body=[
    FunctionDef(
        name='add',
        args=arguments(
            args=[
                arg(arg='self', annotation=None),
                arg(arg='arg1', annotation=None),
                arg(arg='arg2', annotation=None),
                arg(arg='arg3', annotation=None)
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=[
            Assign(targets=[Attribute(value=Name(id='self', ctx=Load()), attr='arg1', ctx=Store())], value=Name(id='arg1', ctx=Load())),
            Assign(targets=[Attribute(value=Name(id='self', ctx=Load()), attr='arg2', ctx=Store())], value=Name(id='arg2', ctx=Load())),
            Assign(targets=[Attribute(value=Name(id='self', ctx=Load()), attr='arg3', ctx=Store())], value=Name(id='arg3', ctx=Load()))
        ],
        decorator_list=[],
        returns=None)
]
)
