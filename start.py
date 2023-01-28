from fractions import Fraction
from dataclasses import dataclass,field
from typing import Optional, NewType, Mapping

# A minimal example to illustrate typechecking.

@dataclass
class NumType:
    pass

@dataclass
class BoolType:
    pass

@dataclass
class StringType:
    pass

SimType = NumType | BoolType

@dataclass
class NumLiteral:
    value: Fraction
    type: SimType = NumType
    def __init__(self, *args):
        self.value = Fraction(*args)

@dataclass
class BoolLiteral:
    value: bool
    type: SimType =BoolType

@dataclass
class StringLiteral:
    value: str
    type: SimType=StringType
    
@dataclass
class Un_boolify:
    left:'AST'

@dataclass
class BinOp:
    operator: str
    left: 'AST'
    right: 'AST'
    type: Optional[SimType] = None

@dataclass
class Variable:
    name: str

@dataclass
class UnOp:
    operator: str
    vari : int

@dataclass
class StringOp:
    operator:str
    left:'AST'
    right:Optional['AST']=None
    #type:StringLiteral

@dataclass
class StringSlice(StringOp):
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None
    type: Optional[SimType] = StringType

@dataclass
class Let:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'

@dataclass
class IfElse:
    condition: 'AST'
    iftrue: 'AST'
    iffalse: 'AST'
    type: Optional[SimType] = None

AST = NumLiteral | BoolLiteral | BinOp | IfElse | StringLiteral


TypedAST = NewType('TypedAST', AST)

class TypeError(Exception):
    pass

# Since we don't have variables, environment is not needed.
def typecheck(program: AST, env = None) -> TypedAST:
    match program:
        case NumLiteral() as t: # already typed.
            return t
        case BoolLiteral() as t: # already typed.
            return t
        case StringLiteral() as t:
            return t
        case BinOp(op, left, right) if op in ["+", "*"]:
            tleft = typecheck(left)
            tright = typecheck(right)
            
            if tleft.type != NumType or tright.type != NumType:
                print(f"left: {tleft}.type")
                print(f"right: {tright}.type")
                raise TypeError()
            return BinOp(op, left, right, NumType)
        case BinOp("<", left, right):
            tleft = typecheck(left)
            tright = typecheck(right)
            if tleft.type != NumType or tright.type != NumType:
                raise TypeError()
            return BinOp("<", left, right, BoolType)
        case BinOp("=", left, right):
            tleft = typecheck(left)
            tright = typecheck(right)
            if tleft.type != tright.type:
                raise TypeError()
            return BinOp("=", left, right, BoolType)
        case IfElse(c, t, f): # We have to typecheck both branches.
            tc = typecheck(c)
            if tc.type != BoolType:
                raise TypeError()
            tt = typecheck(t)
            tf = typecheck(f)
            if tt.type != tf.type: # Both branches must have the same type.
                raise TypeError()
            return IfElse(tc, tt, tf, tt.type) # The common type becomes the type of the if-else.
        case StringSlice("slice",left,start, stop, step):
            tleft = typecheck(left)
            if tleft.type != StringType:
                raise TypeError()
            return StringSlice("slice", left, start, stop, step, StringType)
        
        case Un_boolify(left):
            tleft=typecheck(left)
            if tleft.type!=NumType or StringType:
                raise TypeError()
            return Un_boolify(left)
    raise TypeError()
Value = Fraction

class InvalidProgram(Exception):
    pass

def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
    if environment is None:
        environment = {}
    match program:
        case NumLiteral(value):
            return value
        case StringLiteral(value):
            return value
        case Variable(name):
            if name in environment:
                return environment[name]
            raise InvalidProgram()
        case Let(Variable(name), e1, e2):
            v1 = eval(e1, environment)
            return eval(e2, environment | { name: v1 })
        case BinOp("+", left, right):
            return eval(left, environment) + eval(right, environment)
        case BinOp("-", left, right):
            return eval(left, environment) - eval(right, environment)
        case BinOp("*", left, right):
            return eval(left, environment) * eval(right, environment)
        case BinOp("/", left, right):
            return eval(left, environment) / eval(right, environment)

        # Bitwise Operators With type checking
        case BinOp("&",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                # print(left_type)
                # print(right_type)
                raise InvalidProgram()
            return int(eval(left,environment)) & int(eval(right,environment))
        case BinOp("|",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                print(left_type)
                print(right_type)
                raise InvalidProgram()
            return int(eval(left,environment)) | int(eval(right,environment))
        case BinOp("^",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                print(left_type)
                print(right_type)
                raise InvalidProgram()
            return int(eval(left,environment)) ^ int(eval(right,environment))
        case BinOp(">>",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                print(left_type)
                print(right_type)
                raise InvalidProgram()
            return int(eval(left,environment)) >> int(eval(right,environment))
        case BinOp("<<",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                print(left_type)
                print(right_type)
                raise InvalidProgram()
            return int(eval(left,environment)) << int(eval(right,environment))

        
        # String Operations
        # implement string typecheck for this
        case StringOp('add',left,right):
            return eval(left)+eval(right)
        case StringOp('length',left):
            return len(eval(left))
        case StringSlice("slice", left,start, stop,step):
            left_value = eval(left, environment)
            return left_value[start:stop:step]
         #unary Operations
        case UnOp('-',vari):
            un=eval(vari)
            un=-un
            return eval(NumLiteral(un))
        case UnOp('++',vari):
            un=eval(vari)
            un=un+1
            return eval(NumLiteral(un))
        case UnOp('--',vari):
            un=eval(vari)
            un=un-1
            return eval(NumLiteral(un))
        case Un_boolify(left):
            left_var=eval(left,environment)
            if left_var==0:
                return bool(left_var)
            elif left_var:
                return bool(left_var)
            elif len(left_var)==0:
                return bool(left_var)
            return bool(left_var)
    raise InvalidProgram()
