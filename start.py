from fractions import Fraction
from dataclasses import dataclass,field
from typing import Optional, NewType, Mapping, List


# A minimal example to illustrate typechecking.

class EndOfStream(Exception):
    pass

@dataclass
class Stream:
    source: str
    pos: int

    def from_string(s):
        return Stream(s, 0)

    def next_char(self):
        if self.pos >= len(self.source):
            raise EndOfStream()
        self.pos = self.pos + 1
        return self.source[self.pos - 1]

    def unget(self):
        assert self.pos > 0
        self.pos = self.pos - 1

# Define the token types.

@dataclass
class Num:
    n: int

@dataclass
class Bool:
    b: bool

@dataclass
class Keyword:
    word: str

@dataclass
class Identifier:
    word: str

@dataclass
class Operator:
    op: str

Token = Num | Bool | Keyword | Identifier | Operator

class EndOfTokens(Exception):
    pass

keywords = "if then else end while do done".split()
symbolic_operators = "+ - × / < > ≤ ≥ = ≠".split()
word_operators = "and or not quot rem".split()
whitespace = " \t\n"

def word_to_token(word):
    if word in keywords:
        return Keyword(word)
    if word in word_operators:
        return Operator(word)
    if word == "True":
        return Bool(True)
    if word == "False":
        return Bool(False)
    return Identifier(word)

class TokenError(Exception):
    pass

@dataclass
class Lexer:
    stream: Stream
    save: Token = None

    def from_stream(s):
        return Lexer(s)

    def next_token(self) -> Token:
        try:
            match self.stream.next_char():
                case c if c in symbolic_operators: return Operator(c)
                case c if c.isdigit():
                    n = int(c)
                    while True:
                        try:
                            c = self.stream.next_char()
                            if c.isdigit():
                                n = n*10 + int(c)
                            else:
                                self.stream.unget()
                                return Num(n)
                        except EndOfStream:
                            return Num(n)
                case c if c.isalpha():
                    s = c
                    while True:
                        try:
                            c = self.stream.next_char()
                            if c.isalpha():
                                s = s + c
                            else:
                                self.stream.unget()
                                return word_to_token(s)
                        except EndOfStream:
                            return word_to_token(s)
                case c if c in whitespace:
                    return self.next_token()
        except EndOfStream:
            raise EndOfTokens

    def peek_token(self) -> Token:
        if self.save is not None:
            return self.save
        self.save = self.next_token()
        return self.save

    def advance(self):
        assert self.save is not None
        self.save = None

    def match(self, expected):
        if self.peek_token() == expected:
            return self.advance()
        raise TokenError()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.next_token()
        except EndOfTokens:
            raise StopIteration

@dataclass
class Parser:
    lexer: Lexer

    def from_lexer(lexer):
        return Parser(lexer)

    def parse_if(self):
        self.lexer.match(Keyword("if"))
        c = self.parse_expr()
        self.lexer.match(Keyword("then"))
        t = self.parse_expr()
        self.lexer.match(Keyword("else"))
        f = self.parse_expr()
        self.lexer.match(Keyword("end"))
        return IfElse(c, t, f)

    def parse_while(self):
        self.lexer.match(Keyword("while"))
        c = self.parse_expr()
        self.lexer.match(Keyword("do"))
        b = self.parse_expr()
        self.lexer.match(Keyword("done"))
        return While(c, b)

    def parse_atom(self):
        match self.lexer.peek_token():
            case Identifier(name):
                self.lexer.advance()
                return Variable(name)
            case Num(value):
                self.lexer.advance()
                return NumLiteral(value)
            case Bool(value):
                self.lexer.advance()
                return BoolLiteral(value)

    def parse_mult(self):
        left = self.parse_atom()
        while True:
            match self.lexer.peek_token():
                case Operator(op) if op in "×/":
                    self.lexer.advance()
                    m = self.parse_atom()
                    left = BinOp(op, left, m)
                case _:
                    break
        return left

    def parse_add(self):
        left = self.parse_mult()
        while True:
            match self.lexer.peek_token():
                case Operator(op) if op in "+-":
                    self.lexer.advance()
                    m = self.parse_mult()
                    left = BinOp(op, left, m)
                case _:
                    break
        return left

    def parse_cmp(self):
        left = self.parse_add()
        match self.lexer.peek_token():
            case Operator(op) if op in "<>":
                self.lexer.advance()
                right = self.parse_add()
                return BinOp(op, left, right)
        return left

    def parse_simple(self):
        return self.parse_cmp()

    def parse_expr(self):
        match self.lexer.peek_token():
            case Keyword("if"):
                return self.parse_if()
            case Keyword("while"):
                return self.parse_while()
            case _:
                return self.parse_simple()

@dataclass
class NumType:
    pass

@dataclass
class BoolType:
    pass

@dataclass
class StringType:
    pass

@dataclass
class ListType:
    pass


SimType = NumType | BoolType | StringType | ListType

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

@dataclass
class ListLiteral:
    list_val:list

@dataclass
class ListOp:
    operator:str
    left:'AST'
    right:Optional['AST']=None
    assign:Optional['AST']=None
    type:SimType=ListType

@dataclass
class Get:
    var: 'AST'

@dataclass
class Put:
    var: 'AST'
    e1: 'AST'

@dataclass
class LetMut:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'

@dataclass
class Seq:
    things: List['AST']

@dataclass
class Whilethen:
    condition: 'AST'
    then_body: 'AST'
    type : Optional[SimType] = None

class Environment:
    envs: List

    def __init__(self):
        self.envs = [{}]

    def enter_scope(self):
        self.envs.append({})

    def exit_scope(self):
        assert self.envs
        self.envs.pop()

    def add(self, name, value):
        assert name not in self.envs[-1]
        self.envs[-1][name] = value

    def get(self, name):
        for env in reversed(self.envs):
            if name in env:
                return env[name]
        raise KeyError()

    def update(self, name, value):
        for env in reversed(self.envs):
            if name in env:
                env[name] = value
                return
        raise KeyError()

AST = NumLiteral | BoolLiteral | BinOp | IfElse | StringLiteral | StringOp|ListLiteral|ListOp| Get | Put |Let | LetMut |Seq | Whilethen

Value = Fraction

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
        case UnOp(left):
            tleft=typecheck(left)
            if tleft.type!=NumType:
                raise TypeError()
            return UnOp(left)
        
    raise TypeError()



class InvalidProgram(Exception):
    pass


def eval(program: AST, environment: Environment = None) -> Value:
    if environment is None:
        environment =Environment()
    
    # Pass environment to all explicitally
    def eval2(program):
        return eval(program, environment)
    
    # always call eval2 for enviornment passing instead of eval(program, envi)
    match program:
        case NumLiteral(value):
            return value
        case StringLiteral(value):
            return value
        case Variable(name):
            return environment.get(name)
        case ListLiteral(value):
            # print(f'values: {value}')
            return value
        case Let(Variable(name), e1, e2):
            v1 = eval2(e1)
            environment.enter_scope()
            environment.add(name,v1)
            v2 = eval2(e2)
            environment.exit_scope()
            return v2
        case LetMut(Variable(name),e1,e2):
            v1 = eval2(e1)
            environment.enter_scope()
            environment.add(name,v1)
            v2 = eval2(e2)
            environment.exit_scope()
            return v2
        case Put(Variable(name),e):
            environment.update(name,eval2(e))
            return environment.get(name)
        case Get(Variable(name)):
            return environment.get(name)
        case Seq(things):
            v = None
            for thing in things:
                v = eval2(thing)
            return v
        case BinOp("+", left, right):
            return eval2(left) + eval2(right)
        case BinOp("-", left, right):
            return eval2(left) - eval2(right)
        case BinOp("*", left, right):
            return eval2(left,) * eval2(right )
        case BinOp("/", left, right):
            if(right==0):
                raise InvalidProgram()
            return  eval2(left ) /  eval2(right )
        case BinOp("//", left, right):
            if(right==0):
                raise InvalidProgram()
            return  eval2(left ) //  eval2(right )
        case BinOp("%", left, right):
            if(right==0):
                raise InvalidProgram()
            return  eval2(left ) %  eval2(right )
        case BinOp("**", left, right):
            return  eval2(left ) **  eval2(right )
        case BinOp("==",left,right):
            return  eval2(left ) ==  eval2(right )
        case BinOp("<",left,right):
            return  eval2(left ) <  eval2(right )
        case BinOp(">",left,right):
            return  eval2(left ) >  eval2(right )
        case BinOp(">=",left,right):
            return  eval2(left ) >=  eval2(right )
        case BinOp("<=",left,right):
            return  eval2(left ) <=  eval2(right )
        case BinOp("!=",left,right):
            return  eval2(left ) !=  eval2(right )
    
        # Bitwise Operators With type checking
        case BinOp("&",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                # print(left_type)
                # print(right_type)
                raise InvalidProgram()
            return int( eval2(left )) & int( eval2(right ))
        case BinOp("|",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                print(left_type)
                print(right_type)
                raise InvalidProgram()
            return int( eval2(left )) | int( eval2(right ))
        case BinOp("^",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                print(left_type) 
                print(right_type)
                raise InvalidProgram()
            return int( eval2(left )) ^ int( eval2(right ))
        case BinOp(">>",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                print(left_type)
                print(right_type)
                raise InvalidProgram()
            return int( eval2(left )) >> int( eval2(right ))
        case BinOp("<<",left,right):
            left_type=typecheck(left).type
            right_type=typecheck(right).type
            
            if(left_type!=NumType or right_type!=NumType):
                print(left_type)
                print(right_type)
                raise InvalidProgram()
            return int( eval2(left )) << int( eval2(right ))
  
        # String Operations
        # implement string typecheck for this
        case StringOp('add',left,right):
            return  eval2(left )+ eval2(right )
        case StringOp('length',left):
            return len( eval2(left ))

        case StringSlice("slice", left,start, stop,step):
            left_value =  eval2(left )
            return left_value[start:stop:step]

        #unary Operations
        case UnOp('-',vari):
            un= eval2(vari )
            un=-un
            return  eval2(NumLiteral(un) )
        case UnOp('++',vari):
            un= eval2(vari )
            un=un+1
            return  eval2(NumLiteral(un) )
        case UnOp('--',vari):
            un= eval2(vari )
            un=un-1
            return  eval2(NumLiteral(un) )
        case UnOp('~',vari):
            un=eval2(vari )
            un=-(un+1)
            return  eval2(NumLiteral(un) )

        # IfElse
        case IfElse(c,l,r):
            # if(typecheck(l)!=typecheck(r)):
            #     return InvalidProgram()
            
            condition_eval= eval2(c)
            # print(typech(c))

            if(condition_eval==True):
                return  eval2(l)
                
            else:
                return  eval2(r)
        
        # List Operations
        case ListOp("append",left,right):
            
            # if(right.type!=left.type):
            #     raise InvalidProgram
            l= eval2(left )
            r= eval2(right )
            if(type(r)==Fraction):
                l.append(int(r))
            else:
                l.append(r)
            return l
            

        case ListOp('length',left):
            return len( eval2(left))

        case ListOp('assign',array,index,assign):
            if(typecheck(assign).type!=array.type or typecheck(index).type!=NumType):
                raise InvalidProgram
            arr= eval2(array)
            
            arr[int( eval2(index))]=int( eval2(assign))
            return arr
        
        case ListOp('remove',array):
            arr= eval2(array)
            arr.pop()
            return arr


        case ListOp('remove',array,index):
            if(index!=NumType):
                raise InvalidProgram
            arr= eval2(arr)

        case ListOp('pop',array,index):
            if(index!=NumType):
                raise InvalidProgram
            arr= eval2(array)
            # temp=arr[int( eval2(index))]
            # arr[int( eval2(index))]=len(arr)
            # arr[int( eval2(len(arr)))]=temp
            # print(arr[int( eval2(index))])

            arr.remove(index)
            return arr
        
        # case Whilethen(condi, body):
        #     # condi_check =
        #     # while(eval2(condi)==True):
        #     #     eval2(body)
        #     #     print(eval2(body))
        #     if(eval2(condi)==True):
        #         eval2(body)
        #         Whilethen(condi,)
            # else :
            #     return
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

def test_not():
    a=Variable("a")
    n=NumLiteral(1)
    u1=UnOp("~",a)
    e=Let(a,n,u1)
    # print(eval(e))
    assert eval(e)==-2
test_not()

# def test_while():
#     a  = Variable("a")
#     n1 = NumLiteral(9)
#     b  = Variable("b")
#     n2 = NumLiteral(2)
#     bin1 = BinOp("-", a, b)
#     e  = Let(a, n1, Let(b, n2, bin1))  
#     bin2 = BinOp(">", a, b)
#     condi  = Let(a, n1, Let(b, n2, bin2))
#     # a = eval(e2)
#     # print(a)
#     condi_Block = Whilethen(condi,e)
#     print(eval(condi_Block))
    
# test_while()

# def test_parse():
#     def parse(string):
#         return Parser.parse_expr (
#             Parser.from_lexer(Lexer.from_stream(Stream.from_string(string)))
#         )
#     # You should parse, evaluate and see whether the expression produces the expected value in your tests.
#     print(parse("if a+b > c×d then a×b - c + d else e×f/g end"))

# test_parse() # Uncomment to see the created ASTs.
