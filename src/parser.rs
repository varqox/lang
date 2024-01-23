use crate::{
    lexer::{self, Token, TokenKind},
    source_code,
};

pub type Span<'lexed> = &'lexed [Token<'lexed>];

type Items<'lexed> = Vec<Item<'lexed>>;

self_cell::self_cell!(
    pub struct Program {
        owner: lexer::Program,
        #[covariant]
        dependent: Items,
    }

    impl {Debug}
);

impl Program {
    pub fn lexed_program(&self) -> &lexer::Program {
        self.borrow_owner()
    }

    pub fn items(&self) -> &Items {
        self.borrow_dependent()
    }
}

#[derive(Debug)]
pub struct Item<'lexed> {
    pub span: Span<'lexed>,
    pub outer_attributes: Vec<OuterAttribute<'lexed>>,
    pub visibility: Visibility,
    pub kind: ItemKind<'lexed>,
}

#[derive(Debug)]
pub struct OuterAttribute<'lexed> {
    pub span: Span<'lexed>,
    pub attribute_tree: AttributeTree<'lexed>,
}

#[derive(Debug)]
pub struct AttributeTree<'lexed> {
    pub span: Span<'lexed>,
    pub name: Identifier<'lexed>,
    pub subtrees: Option<Vec<AttributeTree<'lexed>>>,
}

#[derive(Debug, Clone, Copy)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Debug)]
pub enum ItemKind<'lexed> {
    Module(Module<'lexed>),
    Struct(Struct<'lexed>),
    TupleStruct(TupleStruct<'lexed>),
    Function(Function<'lexed>),
    Use(Use<'lexed>),
    TypeAlias(TypeAlias<'lexed>),
    Constant(Constant<'lexed>),
}

#[derive(Debug)]
pub struct Module<'lexed> {
    pub span: Span<'lexed>,
    pub name: Identifier<'lexed>,
    pub items: Option<Vec<Item<'lexed>>>,
}

#[derive(Debug)]
pub struct TupleStruct<'lexed> {
    pub span: Span<'lexed>,
    pub name: Identifier<'lexed>,
    pub generic_params: Option<Vec<GenericParam<'lexed>>>,
    pub field_types: Vec<TupleStructField<'lexed>>,
}

#[derive(Debug)]
pub struct TupleStructField<'lexed> {
    pub span: Span<'lexed>,
    pub visibility: Visibility,
    pub r#type: Type<'lexed>,
}

#[derive(Debug)]
pub struct Struct<'lexed> {
    pub span: Span<'lexed>,
    pub name: Identifier<'lexed>,
    pub generic_params: Option<Vec<GenericParam<'lexed>>>,
    pub fields: Vec<StructField<'lexed>>,
}

#[derive(Debug)]
pub struct StructField<'lexed> {
    pub span: Span<'lexed>,
    pub visibility: Visibility,
    pub name: Identifier<'lexed>,
    pub r#type: Type<'lexed>,
}

#[derive(Debug)]
pub struct Function<'lexed> {
    pub span: Span<'lexed>,
    pub name: Identifier<'lexed>,
    pub generic_params: Option<Vec<GenericParam<'lexed>>>,
    pub params: Vec<FunctionParam<'lexed>>,
    pub return_type: Option<Type<'lexed>>,
    pub body: BlockExpression<'lexed>,
}

#[derive(Debug)]
pub struct FunctionParam<'lexed> {
    pub span: Span<'lexed>,
    pub mutable: bool,
    pub name: Identifier<'lexed>,
    pub r#type: Type<'lexed>,
}

#[derive(Debug)]
pub struct TypeAlias<'lexed> {
    pub span: Span<'lexed>,
    pub name: Identifier<'lexed>,
    pub generic_params: Option<Vec<GenericParam<'lexed>>>,
    pub r#type: Type<'lexed>,
}

#[derive(Debug)]
pub struct Use<'lexed> {
    pub span: Span<'lexed>,
    pub path: Path<'lexed>,
}

#[derive(Debug)]
pub struct Constant<'lexed> {
    pub span: Span<'lexed>,
    pub name: Identifier<'lexed>,
    pub generic_params: Option<Vec<GenericParam<'lexed>>>,
    pub r#type: Option<Type<'lexed>>,
    pub value: Expression<'lexed>,
}

#[derive(Debug)]
pub struct Type<'lexed> {
    pub span: Span<'lexed>,
    pub kind: TypeKind<'lexed>,
}

#[derive(Debug)]
pub enum TypeKind<'lexed> {
    Never,
    Path(Path<'lexed>),
    Tuple(Vec<Type<'lexed>>),
    Array(Box<Type<'lexed>>, Expression<'lexed>),
    Reference {
        mutable: bool,
        r#type: Box<Type<'lexed>>,
    },
}

#[derive(Debug)]
pub struct Path<'lexed> {
    pub span: Span<'lexed>,
    pub global: bool,
    pub segments: Vec<PathSegment<'lexed>>,
}

#[derive(Debug)]
pub struct PathSegment<'lexed> {
    pub span: Span<'lexed>,
    pub identifier: Identifier<'lexed>,
    pub generic_args: Option<Vec<Expression<'lexed>>>,
}

#[derive(Debug)]
pub struct GenericParam<'lexed> {
    pub span: Span<'lexed>,
    pub kind: GenericParamKind<'lexed>,
}

#[derive(Debug)]
pub enum GenericParamKind<'lexed> {
    Type {
        name: Identifier<'lexed>,
    },
    ConstValue {
        name: Identifier<'lexed>,
        r#type: Type<'lexed>,
    },
}

#[derive(Debug)]
pub struct Expression<'lexed> {
    pub span: Span<'lexed>,
    pub kind: ExpressionKind<'lexed>,
}

#[derive(Debug)]
pub enum ExpressionKind<'lexed> {
    Underscore,
    Continue,
    Break,
    IntegerLiteral(IntegerLiteral<'lexed>),
    ByteLiteral(u8),
    ByteStringLiteral(Vec<u8>),
    Path(Path<'lexed>),
    Tuple(Vec<Expression<'lexed>>),
    ArrayLiteral(Vec<Expression<'lexed>>),
    ArrayWithSize(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    Block(Box<BlockExpression<'lexed>>),
    Return(Box<Expression<'lexed>>),
    BoolNegation(Box<Expression<'lexed>>),
    ArithmeticNegation(Box<Expression<'lexed>>),
    // Assign
    Assign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    PlusAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    MinusAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    MultiplyAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    DivideAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    DivideRemainderAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LogicOrAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LogicAndAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LogicXorAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LogicShiftLeftAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LogicShiftRightAssign(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    // Logic
    LogicOr(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LogicAnd(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LogicXor(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LogicShiftLeft(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LogicShiftRight(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    // Arithmetic
    Plus(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    Minus(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    Multiply(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    Divide(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    DivideRemainter(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    // Compare
    CompareEqual(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    CompareNotEqual(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    CompareGreater(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    CompareGreaterEqual(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    CompareLess(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    CompareLessEqual(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    // Lazy bool
    LazyBoolAnd(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    LazyBoolOr(Box<Expression<'lexed>>, Box<Expression<'lexed>>),
    // Reference
    TakeReference(Box<Expression<'lexed>>),
    TakeMutReference(Box<Expression<'lexed>>),
    Dereference(Box<Expression<'lexed>>),
    // Other
    Call {
        callee: Box<Expression<'lexed>>,
        args: Vec<Expression<'lexed>>,
    },
    Cast {
        expr: Box<Expression<'lexed>>,
        r#type: Box<Type<'lexed>>,
    },
    MemberExpression {
        obj: Box<Expression<'lexed>>,
        member: Member<'lexed>,
    },
    If(Box<IfExpression<'lexed>>),
    Const(Box<Expression<'lexed>>),
    Loop(Box<BlockExpression<'lexed>>),
    WhileLoop {
        condition: Box<Expression<'lexed>>,
        body: Box<BlockExpression<'lexed>>,
    },
}

#[derive(Debug)]
pub enum Member<'lexed> {
    Identifier(Identifier<'lexed>),
    IntegerLiteral(IntegerLiteral<'lexed>),
}

#[derive(Debug)]
pub struct IfExpression<'lexed> {
    pub span: Span<'lexed>,
    pub condition: Expression<'lexed>,
    pub true_branch: BlockExpression<'lexed>,
    pub r#else: Option<ElseExpression<'lexed>>,
}

#[derive(Debug)]
pub enum ElseExpression<'lexed> {
    If(Box<IfExpression<'lexed>>),
    Block(BlockExpression<'lexed>),
}

#[derive(Debug)]
pub struct BlockExpression<'lexed> {
    pub span: Span<'lexed>,
    pub statements: Vec<Statement<'lexed>>,
    pub final_expression: Option<Expression<'lexed>>,
}

#[derive(Debug)]
pub struct Statement<'lexed> {
    pub span: Span<'lexed>,
    pub kind: StatementKind<'lexed>,
}

#[derive(Debug)]
pub enum StatementKind<'lexed> {
    Expression(Expression<'lexed>),
    Let {
        mutable: bool,
        name: Identifier<'lexed>,
        r#type: Option<Type<'lexed>>,
        value: Expression<'lexed>,
    },
    Const {
        name: Identifier<'lexed>,
        r#type: Option<Type<'lexed>>,
        value: Expression<'lexed>,
    },
}

#[derive(Debug)]
pub struct Identifier<'lexed> {
    pub span: Span<'lexed>,
    pub name: &'lexed str,
}

#[derive(Debug)]
pub struct IntegerLiteral<'lexed> {
    span: Span<'lexed>,
    value: String,
    suffix: &'lexed str,
}

struct Parser<'lexed> {
    source_code: &'lexed source_code::Program,
    lexed_program: &'lexed lexer::Program,
}

struct State {
    current_token_idx: usize,
}

impl State {
    fn start_span(&self) -> SpanStart {
        SpanStart {
            token_idx: self.current_token_idx,
        }
    }
}

struct SpanStart {
    token_idx: usize,
}

fn slices_meet(a: &str, b: &str) -> bool {
    a[a.len()..].as_ptr() == b.as_ptr()
}

#[derive(Debug)]
struct ExtractedSequence<T> {
    has_trailing_separator: bool,
    elems: Vec<T>,
}

#[derive(Debug)]
enum StructOrTupleStruct<'lexed> {
    Struct(Struct<'lexed>),
    TupleStruct(TupleStruct<'lexed>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum ExpressionOperator {
    Return,
    Equal,
    PlusEqual,
    MinusEqual,
    StarEqual,
    SlashEqual,
    PercentEqual,
    LogicOrEqual,
    LogicAndEqual,
    LogicXorEqual,
    LogicShiftLeftEqual,
    LogicShiftRightEqual,
    LogicOr,
    LogicAnd,
    LogicXor,
    LogicShiftLeft,
    LogicShiftRight,
    LazyBoolOr,
    LazyBoolAnd,
    Less,
    Greater,
    EqualEqual,
    NotEqual,
    LessEqual,
    GreaterEqual,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    CastAs,
    Const,
    TakeReference,
    TakeMutReference,
    Dereference,
    BoolNegation,
    ArithmeticNegation,
    Dot,
    Call,
}

#[derive(Debug, PartialEq, Eq)]
enum ExpressionOperatorAssociativity {
    Left,  // ... op ... op ... <=> (... op ...) op ...
    Right, // ... op ... op ... <=> ... op (... op ...)
    None,  // ... op ... op ... is disallowed
}

#[derive(Debug)]
struct ExpressionOperatorInfo {
    priority: u8,
    associativity: ExpressionOperatorAssociativity,
    // Whether to allow e.g. ... && ... || ... or not
    associates_with_other_of_the_same_priority: bool,
}

impl ExpressionOperator {
    fn to_info(&self) -> ExpressionOperatorInfo {
        use ExpressionOperatorAssociativity::Left as AssocLeft;
        use ExpressionOperatorAssociativity::None as AssocNone;
        use ExpressionOperatorAssociativity::Right as AssocRight;
        let op_info = |priority, associativity, associates_with_other_of_the_same_priority| {
            ExpressionOperatorInfo {
                priority,
                associativity,
                associates_with_other_of_the_same_priority,
            }
        };
        match self {
            Self::Return => op_info(0, AssocNone, false),
            Self::Equal
            | Self::PlusEqual
            | Self::MinusEqual
            | Self::StarEqual
            | Self::SlashEqual
            | Self::PercentEqual
            | Self::LogicOrEqual
            | Self::LogicAndEqual
            | Self::LogicXorEqual
            | Self::LogicShiftLeftEqual
            | Self::LogicShiftRightEqual => op_info(1, AssocNone, false),
            Self::LogicOr
            | Self::LogicAnd
            | Self::LogicXor
            | Self::LogicShiftLeft
            | Self::LogicShiftRight => op_info(2, AssocLeft, false),
            Self::LazyBoolOr => op_info(3, AssocRight, false),
            Self::LazyBoolAnd => op_info(4, AssocRight, false),
            Self::Less
            | Self::Greater
            | Self::EqualEqual
            | Self::NotEqual
            | Self::LessEqual
            | Self::GreaterEqual => op_info(5, AssocNone, false),
            Self::Plus | Self::Minus => op_info(6, AssocLeft, true),
            Self::Star | Self::Slash | Self::Percent => op_info(7, AssocLeft, true),
            Self::CastAs => op_info(8, AssocLeft, false),
            Self::Const => op_info(9, AssocRight, false),
            Self::TakeReference
            | Self::TakeMutReference
            | Self::Dereference
            | Self::ArithmeticNegation
            | Self::BoolNegation => op_info(10, AssocNone, false),
            Self::Dot | Self::Call => op_info(11, AssocLeft, true),
        }
    }

    fn to_print_string(&self) -> &'static str {
        match self {
            Self::Return => "return",
            Self::Equal => "=",
            Self::PlusEqual => "+=",
            Self::MinusEqual => "-=",
            Self::StarEqual => "*=",
            Self::SlashEqual => "/=",
            Self::PercentEqual => "%=",
            Self::LogicOrEqual => "|=",
            Self::LogicAndEqual => "&=",
            Self::LogicXorEqual => "^=",
            Self::LogicShiftLeftEqual => "<<=",
            Self::LogicShiftRightEqual => ">>=",
            Self::LogicOr => "|",
            Self::LogicAnd => "&",
            Self::LogicXor => "^",
            Self::LogicShiftLeft => "<<",
            Self::LogicShiftRight => ">>",
            Self::LazyBoolOr => "||",
            Self::LazyBoolAnd => "&&",
            Self::Less => "<",
            Self::Greater => ">",
            Self::EqualEqual => "==",
            Self::NotEqual => "!=",
            Self::LessEqual => "<=",
            Self::GreaterEqual => ">=",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Star => "*",
            Self::Slash => "/",
            Self::Percent => "%",
            Self::CastAs => "as",
            Self::Const => "const",
            Self::TakeReference => "&",
            Self::TakeMutReference => "&mut",
            Self::Dereference => "*",
            Self::BoolNegation => "!",
            Self::ArithmeticNegation => "-",
            Self::Dot => ".",
            Self::Call => panic!("BUG: should not be called"),
        }
    }
}
#[derive(Debug)]
struct ExpressionOperatorWithLeftExpr<'lexed> {
    operator_span: Span<'lexed>,
    operator: ExpressionOperator,
    left_expr: Option<Expression<'lexed>>,
}

impl<'lexed> Parser<'lexed> {
    fn end_span(&self, state: &State, span_start: &SpanStart) -> Span<'lexed> {
        &self.lexed_program.tokens()[span_start.token_idx..state.current_token_idx]
    }

    fn overspan(&self, a: &Span, b: &Span) -> Span<'lexed> {
        let base = self.lexed_program.tokens()[..].as_ptr() as usize;
        let start = a.as_ptr() as usize;
        let end = b[b.len()..].as_ptr() as usize;
        assert!(start >= base);
        assert!(end >= base);
        const ELEM_SIZE: usize = std::mem::size_of::<Token>();
        &self.lexed_program.tokens()[((start - base) / ELEM_SIZE)..((end - base) / ELEM_SIZE)]
    }

    fn missing_token(&self, state: &State, missing_tokens: core::fmt::Arguments) -> ! {
        let slice = if state.current_token_idx == self.lexed_program.tokens().len() {
            &self.source_code.code.as_str()[self.source_code.code.len()..]
        } else if state.current_token_idx == 0 {
            &self.source_code.code.as_str()[0..0]
        } else {
            let slice = self.lexed_program.tokens()[state.current_token_idx - 1].slice;
            &slice[slice.len()..]
        };
        self.source_code.error_on(
            &slice,
            format_args!("missing token: {}", missing_tokens),
            format_args!("missing {}", missing_tokens),
            &mut [].into_iter(),
        );
    }

    fn unexpected_eof(&self, expected: core::fmt::Arguments) -> ! {
        self.source_code.error_on(
            &self.source_code.code[self.source_code.code.len()..],
            format_args!("unexpected end of file, expected {}", expected),
            format_args!("expected {}", expected),
            &mut [].into_iter(),
        );
    }

    fn unexpected_token(&self, state: &State, expected: core::fmt::Arguments) -> ! {
        let token = &self.lexed_program.tokens()[state.current_token_idx];
        self.source_code.error_on(
            token.slice,
            format_args!("unexpected token {}, expected {}", token.kind, expected),
            format_args!("expected {}", expected),
            &mut [].into_iter(),
        );
    }

    fn peek_token(&self, state: &State) -> Option<&'lexed Token<'lexed>> {
        self.peek_token_at(state, 0)
    }

    fn peek_token_at(&self, state: &State, idx: usize) -> Option<&'lexed Token<'lexed>> {
        self.lexed_program
            .tokens()
            .get(state.current_token_idx + idx)
    }

    fn extract_token(&self, state: &mut State) -> &'lexed Token<'lexed> {
        let token = self.peek_token(state).unwrap();
        state.current_token_idx += 1;
        token
    }

    fn extract_token_of_kind(&self, state: &mut State, kind: TokenKind) -> &'lexed Token<'lexed> {
        let Some(token) = self.peek_token(state) else {
            self.unexpected_eof(format_args!("{}", kind));
        };
        if token.kind == kind {
            state.current_token_idx += 1;
            return token;
        }
        self.unexpected_token(state, format_args!("{}", kind));
    }

    fn extract_identifier(&self, state: &mut State) -> Identifier<'lexed> {
        let span_start = state.start_span();
        let ident = self.extract_token_of_kind(state, TokenKind::Identifier);
        Identifier {
            span: self.end_span(state, &span_start),
            name: ident.slice,
        }
    }

    fn extract_integer_literal(&self, state: &mut State) -> IntegerLiteral<'lexed> {
        let span_start = state.start_span();
        let integer = self.extract_token_of_kind(state, TokenKind::Integer);
        let mut value = String::new();
        let mut suffix = &integer.slice[integer.slice.len()..];
        {
            for (idx, c) in integer.slice.char_indices() {
                if c == '_' {
                    continue;
                }
                if c.is_digit(10) {
                    value.push(c);
                } else {
                    suffix = &integer.slice[idx..];
                    break;
                }
            }
        }
        assert!(!value.is_empty());
        IntegerLiteral {
            span: self.end_span(state, &span_start),
            value,
            suffix,
        }
    }

    fn extract_byte_or_bytes_literal(&self, state: &mut State, kind: TokenKind) -> Vec<u8> {
        assert!(kind == TokenKind::Byte || kind == TokenKind::ByteString);
        let token = self.extract_token_of_kind(state, kind);
        let mut bytes = Vec::new();
        let mut token_chars = token.slice[..token.slice.len() - 1].char_indices().skip(2);
        while let Some((idx, c)) = token_chars.next() {
            if c == '\\' {
                let Some((idx1, c1)) = token_chars.next() else {
                    self.source_code.error_on(
                        &token.slice[idx + c.len_utf8()..idx + c.len_utf8()],
                        format_args!("missing escaped character"),
                        format_args!("missing escaped character"),
                        &mut [].into_iter(),
                    );
                };
                match c1 {
                    '\\' => bytes.push(b'\\'),
                    '\'' => bytes.push(b'\''),
                    '"' => bytes.push(b'"'),
                    '0' => bytes.push(0),
                    'r' => bytes.push(b'\r'),
                    'n' => bytes.push(b'\n'),
                    't' => bytes.push(b'\t'),
                    'x' => {
                        let Some((idx2, c2)) = token_chars.next() else {
                            self.source_code.error_on(
                                &token.slice[idx1 + c1.len_utf8()..idx1 + c1.len_utf8()],
                                format_args!("missing hex digit"),
                                format_args!("missing hex digit"),
                                &mut [].into_iter(),
                            );
                        };
                        let Some(d1) = c2.to_digit(16) else {
                            self.source_code.error_on(
                                &token.slice[idx2..idx2 + c2.len_utf8()],
                                format_args!("invalid hex digit in escape sequence"),
                                format_args!("invalid hex digit"),
                                &mut [].into_iter(),
                            );
                        };
                        let Some((idx3, c3)) = token_chars.next() else {
                            self.source_code.error_on(
                                &token.slice[idx2 + c2.len_utf8()..idx2 + c2.len_utf8()],
                                format_args!("missing hex digit"),
                                format_args!("missing hex digit"),
                                &mut [].into_iter(),
                            );
                        };
                        let Some(d2) = c3.to_digit(16) else {
                            self.source_code.error_on(
                                &token.slice[idx3..idx3 + c3.len_utf8()],
                                format_args!("invalid hex digit in escape sequence"),
                                format_args!("invalid hex digit"),
                                &mut [].into_iter(),
                            );
                        };
                        bytes.push(((d1 as u8) << 4) | (d2 as u8));
                    }
                    _ => {
                        self.source_code.error_on(
                            &token.slice[idx1..idx1 + c1.len_utf8()],
                            format_args!("invalid escape sequence in byte string literal"),
                            format_args!("invalid escape sequence"),
                            &mut [].into_iter(),
                        );
                    }
                }
                continue;
            }
            if !c.is_ascii() {
                self.source_code.error_on(
                    &token.slice[idx..idx + c.len_utf8()],
                    format_args!("non-ASCII character in byte literal"),
                    format_args!("non-ASCII character in byte literal"),
                    &mut [].into_iter(),
                );
            }
            bytes.push(c as u8);
        }
        bytes
    }

    fn extract_sequence_with_optional_trailing_separator_using_function<T>(
        &self,
        state: &mut State,
        start_token_kind: TokenKind,
        extractor: fn(&Self, &mut State) -> T,
        extracted_node_name_str: &'static str,
        separator: TokenKind,
        end_token_kind: TokenKind,
    ) -> ExtractedSequence<T> {
        self.extract_token_of_kind(state, start_token_kind);
        let mut elems = Vec::new();
        let mut has_trailing_separator = false;
        loop {
            match self.peek_token(state) {
                None => self.unexpected_eof(format_args!(
                    "{} or {}",
                    end_token_kind, extracted_node_name_str
                )),
                Some(token) if token.kind == end_token_kind => {
                    self.extract_token(state);
                    return ExtractedSequence {
                        has_trailing_separator,
                        elems,
                    };
                }
                _ => {
                    elems.push(extractor(self, state));
                    has_trailing_separator = false;
                    match self.peek_token(state) {
                        None => {
                            self.unexpected_eof(format_args!("{} or {}", end_token_kind, separator))
                        }
                        Some(token) => match token.kind {
                            kind if kind == end_token_kind => {
                                self.extract_token(state);
                                return ExtractedSequence {
                                    has_trailing_separator,
                                    elems,
                                };
                            }
                            kind if kind == separator => {
                                self.extract_token(state);
                                has_trailing_separator = true;
                            }
                            _ => self.missing_token(
                                state,
                                format_args!("{} or {}", end_token_kind, separator),
                            ),
                        },
                    }
                }
            }
        }
    }

    fn extract_type(&self, state: &mut State) -> Type<'lexed> {
        let span_start = state.start_span();
        let Some(token) = self.peek_token(state) else {
            self.unexpected_eof(format_args!("type"));
        };
        match token.kind {
            TokenKind::Exclamation => {
                self.extract_token(state);
                Type {
                    span: self.end_span(state, &span_start),
                    kind: TypeKind::Never,
                }
            }
            TokenKind::ColonColon | TokenKind::Identifier => {
                let path = self.extract_path(state);
                Type {
                    span: self.end_span(state, &span_start),
                    kind: TypeKind::Path(path),
                }
            }
            TokenKind::Ampersand => {
                self.extract_token(state);
                let mutable = match self.peek_token(state) {
                    None => self.unexpected_eof(format_args!("mut or type")),
                    Some(token) if token.kind == TokenKind::KeywordMut => {
                        self.extract_token(state);
                        true
                    }
                    _ => false,
                };
                let r#type = self.extract_type(state);
                Type {
                    span: self.end_span(state, &span_start),
                    kind: TypeKind::Reference {
                        mutable,
                        r#type: Box::new(r#type),
                    },
                }
            }
            TokenKind::Lsquare => {
                self.extract_token(state);
                let r#type = self.extract_type(state);
                self.extract_token_of_kind(state, TokenKind::Semicolon);
                let size_expr = self.extract_expression(state, &[TokenKind::Rsquare]);
                self.extract_token_of_kind(state, TokenKind::Rsquare);
                Type {
                    span: self.end_span(state, &span_start),
                    kind: TypeKind::Array(Box::new(r#type), size_expr),
                }
            }
            TokenKind::Lparen => {
                let ExtractedSequence { elems, .. } = self
                    .extract_sequence_with_optional_trailing_separator_using_function(
                        state,
                        TokenKind::Lparen,
                        Self::extract_type,
                        "type",
                        TokenKind::Comma,
                        TokenKind::Rparen,
                    );
                Type {
                    span: self.end_span(state, &span_start),
                    kind: TypeKind::Tuple(elems),
                }
            }
            _ => self.unexpected_token(state, format_args!("type")),
        }
    }

    fn extract_path(&self, state: &mut State) -> Path<'lexed> {
        let span_start = state.start_span();
        let global = match self.peek_token(state) {
            None => self.unexpected_eof(format_args!(":: or identifier")),
            Some(token) if token.kind == TokenKind::ColonColon => true,
            _ => false,
        };
        let mut segments = Vec::new();
        loop {
            let span_start = state.start_span();
            let identifier = self.extract_identifier(state);
            let Some(token) = self.peek_token(state) else {
                segments.push(PathSegment {
                    span: self.end_span(state, &span_start),
                    identifier,
                    generic_args: None,
                });
                break;
            };
            match token.kind {
                TokenKind::Langle => {
                    let args = self
                        .extract_sequence_with_optional_trailing_separator_using_function(
                            state,
                            TokenKind::Langle,
                            |this: &Parser, state: &mut State| {
                                this.extract_expression(
                                    state,
                                    &[TokenKind::Rangle, TokenKind::Comma],
                                )
                            },
                            "expression",
                            TokenKind::Comma,
                            TokenKind::Rangle,
                        )
                        .elems;

                    segments.push(PathSegment {
                        span: self.end_span(state, &span_start),
                        identifier,
                        generic_args: Some(args),
                    });
                    match self.peek_token(state) {
                        Some(token) if token.kind == TokenKind::ColonColon => {
                            self.extract_token(state);
                        }
                        _ => break,
                    }
                }
                _ => {
                    segments.push(PathSegment {
                        span: self.end_span(state, &span_start),
                        identifier,
                        generic_args: None,
                    });
                    if token.kind == TokenKind::ColonColon {
                        self.extract_token(state);
                    } else {
                        break;
                    }
                }
            }
        }

        Path {
            span: self.end_span(state, &span_start),
            global,
            segments,
        }
    }

    fn expression_parser_fold_stack(
        &self,
        stack: &mut Vec<ExpressionOperatorWithLeftExpr<'lexed>>,
        expr: Expression<'lexed>,
        next_op: Option<(ExpressionOperator, &Span<'lexed>)>,
    ) -> Expression<'lexed> {
        let next_op_info = next_op.as_ref().map(|(operator, _)| operator.to_info());
        let mut expr = Some(expr);
        while !stack.is_empty() {
            let last_op = stack.last().unwrap();
            let last_op_operator = last_op.operator.clone();
            let last_op_operator_span = last_op.operator_span;
            let last_op_info = last_op_operator.to_info();
            let mut fold = || {
                let ExpressionOperatorWithLeftExpr {
                    operator_span,
                    operator,
                    left_expr,
                } = stack.pop().unwrap();
                macro_rules! unary_op {
                    ($expression_kind:ident) => {{
                        expr = Some(Expression {
                            span: self.overspan(&operator_span, &expr.as_ref().unwrap().span),
                            kind: ExpressionKind::$expression_kind(Box::new(expr.take().unwrap())),
                        })
                    }};
                }
                macro_rules! binary_op {
                    ($expression_kind:ident) => {{
                        let left_expr = left_expr.unwrap();
                        let right_expr = expr.take().unwrap();
                        expr = Some(Expression {
                            span: self.overspan(&left_expr.span, &right_expr.span),
                            kind: ExpressionKind::$expression_kind(
                                Box::new(left_expr),
                                Box::new(right_expr),
                            ),
                        })
                    }};
                }

                match operator {
                    ExpressionOperator::ArithmeticNegation => unary_op!(ArithmeticNegation),
                    ExpressionOperator::BoolNegation => unary_op!(BoolNegation),
                    ExpressionOperator::Const => unary_op!(Const),
                    ExpressionOperator::Dereference => unary_op!(Dereference),
                    ExpressionOperator::Equal => binary_op!(Assign),
                    ExpressionOperator::EqualEqual => binary_op!(CompareEqual),
                    ExpressionOperator::Greater => binary_op!(CompareGreater),
                    ExpressionOperator::GreaterEqual => binary_op!(CompareGreaterEqual),
                    ExpressionOperator::LazyBoolAnd => binary_op!(LazyBoolAnd),
                    ExpressionOperator::LazyBoolOr => binary_op!(LazyBoolOr),
                    ExpressionOperator::Less => binary_op!(CompareLess),
                    ExpressionOperator::LessEqual => binary_op!(CompareLessEqual),
                    ExpressionOperator::LogicAnd => binary_op!(LogicAnd),
                    ExpressionOperator::LogicAndEqual => binary_op!(LogicAndAssign),
                    ExpressionOperator::LogicOr => binary_op!(LogicOr),
                    ExpressionOperator::LogicOrEqual => binary_op!(LogicOrAssign),
                    ExpressionOperator::LogicShiftLeft => binary_op!(LogicShiftLeft),
                    ExpressionOperator::LogicShiftLeftEqual => binary_op!(LogicShiftLeftAssign),
                    ExpressionOperator::LogicShiftRight => binary_op!(LogicShiftRight),
                    ExpressionOperator::LogicShiftRightEqual => binary_op!(LogicShiftRightAssign),
                    ExpressionOperator::LogicXor => binary_op!(LogicXor),
                    ExpressionOperator::LogicXorEqual => binary_op!(LogicXorAssign),
                    ExpressionOperator::Minus => binary_op!(Minus),
                    ExpressionOperator::MinusEqual => binary_op!(MinusAssign),
                    ExpressionOperator::NotEqual => binary_op!(CompareNotEqual),
                    ExpressionOperator::Percent => binary_op!(DivideRemainter),
                    ExpressionOperator::PercentEqual => binary_op!(DivideRemainderAssign),
                    ExpressionOperator::Plus => binary_op!(Plus),
                    ExpressionOperator::PlusEqual => binary_op!(PlusAssign),
                    ExpressionOperator::Return => unary_op!(Return),
                    ExpressionOperator::Slash => binary_op!(Divide),
                    ExpressionOperator::SlashEqual => binary_op!(DivideAssign),
                    ExpressionOperator::Star => binary_op!(Multiply),
                    ExpressionOperator::StarEqual => binary_op!(MultiplyAssign),
                    ExpressionOperator::TakeMutReference => unary_op!(TakeMutReference),
                    ExpressionOperator::TakeReference => unary_op!(TakeReference),
                    ExpressionOperator::CastAs
                    | ExpressionOperator::Dot
                    | ExpressionOperator::Call => {
                        panic!("BUG: {:?} operator should not be on the stack", operator)
                    }
                }
            };

            let Some(next_op_info) = &next_op_info else {
                fold();
                continue;
            };
            if last_op_info.priority > next_op_info.priority {
                fold();
                continue;
            }
            if last_op_info.priority == next_op_info.priority {
                assert!(last_op_info.associativity == next_op_info.associativity);
                assert!(
                    last_op_info.associates_with_other_of_the_same_priority
                        == next_op_info.associates_with_other_of_the_same_priority
                );
                let (next_operator, next_op_span) = next_op.as_ref().unwrap();
                if &last_op_operator != next_operator
                    && !last_op_info.associates_with_other_of_the_same_priority
                {
                    self.lexed_program.error_on(
                        &self.overspan(&last_op_operator_span, &next_op_span),
                        format_args!(
                            "combining {} and {} without parenthesis is disallowed",
                            last_op_operator.to_print_string(),
                            next_operator.to_print_string()
                        ),
                        format_args!(
                            "cannot combine these operators, use parenthesis around one of them"
                        ),
                        &mut [].into_iter(),
                    );
                }
                match last_op_info.associativity {
                    ExpressionOperatorAssociativity::None => self.lexed_program.error_on(
                        &self.overspan(&last_op_operator_span, &next_op_span),
                        format_args!(
                            "combining {} and {} without parenthesis is disallowed",
                            last_op_operator.to_print_string(),
                            next_operator.to_print_string()
                        ),
                        format_args!(
                            "cannot combine these operators, use parenthesis around one of them"
                        ),
                        &mut [].into_iter(),
                    ),
                    ExpressionOperatorAssociativity::Left => {
                        fold();
                        continue;
                    }
                    ExpressionOperatorAssociativity::Right => {
                        break;
                    }
                }
            }
            break;
        }
        expr.unwrap()
    }

    fn expression_parser_extract_simple_expression(
        &self,
        state: &mut State,
        stack: &mut Vec<ExpressionOperatorWithLeftExpr<'lexed>>,
    ) -> Expression<'lexed> {
        loop {
            let expr_span_start = state.start_span();
            let Some(token) = self.peek_token(state) else {
                self.unexpected_eof(format_args!("expression"))
            };
            match token.kind {
                TokenKind::Underscore => {
                    self.extract_token(state);
                    break Expression {
                        span: self.end_span(state, &expr_span_start),
                        kind: ExpressionKind::Underscore,
                    };
                }
                TokenKind::KeywordContinue => {
                    self.extract_token(state);
                    break Expression {
                        span: self.end_span(state, &expr_span_start),
                        kind: ExpressionKind::Continue,
                    };
                }
                TokenKind::KeywordBreak => {
                    break Expression {
                        span: self.end_span(state, &expr_span_start),
                        kind: ExpressionKind::Break,
                    };
                }
                TokenKind::Integer => {
                    let integer_literal = self.extract_integer_literal(state);
                    break Expression {
                        span: self.end_span(state, &expr_span_start),
                        kind: ExpressionKind::IntegerLiteral(integer_literal),
                    };
                }
                TokenKind::Byte => {
                    let bytes = self.extract_byte_or_bytes_literal(state, TokenKind::Byte);
                    let span = self.end_span(state, &expr_span_start);
                    match bytes.len() {
                        0 => self.lexed_program.error_on(
                            span,
                            format_args!("empty byte literal"),
                            format_args!("empty byte literal"),
                            &mut [].into_iter(),
                        ),
                        1 => {
                            break Expression {
                                span,
                                kind: ExpressionKind::ByteLiteral(bytes[0]),
                            }
                        }
                        _ => self.lexed_program.error_on(
                            span,
                            format_args!("byte literal has to have length 1"),
                            format_args!("byte literal has to have length 1"),
                            &mut [].into_iter(),
                        ),
                    }
                }
                TokenKind::ByteString => {
                    let bytes = self.extract_byte_or_bytes_literal(state, TokenKind::ByteString);
                    break Expression {
                        span: self.end_span(state, &expr_span_start),
                        kind: ExpressionKind::ByteStringLiteral(bytes),
                    };
                }
                TokenKind::Identifier => {
                    let ident = self.extract_identifier(state);
                    break Expression {
                        span: ident.span,
                        kind: ExpressionKind::Path(Path {
                            span: ident.span,
                            global: false,
                            segments: vec![PathSegment {
                                span: ident.span,
                                identifier: ident,
                                generic_args: None,
                            }],
                        }),
                    };
                }
                TokenKind::ColonColon => {
                    self.extract_token(state);
                    let ident = self.extract_identifier(state);
                    let span = self.end_span(state, &expr_span_start);
                    break Expression {
                        span,
                        kind: ExpressionKind::Path(Path {
                            span,
                            global: true,
                            segments: vec![PathSegment {
                                span: ident.span,
                                identifier: ident,
                                generic_args: None,
                            }],
                        }),
                    };
                }
                TokenKind::Lparen => {
                    let extracted_sequence = self
                        .extract_sequence_with_optional_trailing_separator_using_function(
                            state,
                            TokenKind::Lparen,
                            |this: &Parser, state: &mut State| {
                                this.extract_expression(
                                    state,
                                    &[TokenKind::Rparen, TokenKind::Comma],
                                )
                            },
                            "expression",
                            TokenKind::Comma,
                            TokenKind::Rparen,
                        );
                    let span = self.end_span(state, &expr_span_start);
                    break if extracted_sequence.elems.len() == 1
                        && !extracted_sequence.has_trailing_separator
                    {
                        extracted_sequence.elems.into_iter().next().unwrap()
                    } else {
                        Expression {
                            span,
                            kind: ExpressionKind::Tuple(extracted_sequence.elems),
                        }
                    };
                }
                TokenKind::Lsquare => {
                    self.extract_token(state);
                    let expr = self.extract_expression(
                        state,
                        &[TokenKind::Rsquare, TokenKind::Semicolon, TokenKind::Comma],
                    );
                    break match self.peek_token(state) {
                        None => self.unexpected_eof(format_args!("] or ; or ,")),
                        Some(token) if token.kind == TokenKind::Rsquare => {
                            self.extract_token(state);
                            Expression {
                                span: self.end_span(state, &expr_span_start),
                                kind: ExpressionKind::ArrayLiteral(vec![expr]),
                            }
                        }
                        Some(token) if token.kind == TokenKind::Semicolon => {
                            self.extract_token(state);
                            let size_expr = self.extract_expression(state, &[TokenKind::Rsquare]);
                            self.extract_token_of_kind(state, TokenKind::Rsquare);
                            Expression {
                                span: self.end_span(state, &expr_span_start),
                                kind: ExpressionKind::ArrayWithSize(
                                    Box::new(expr),
                                    Box::new(size_expr),
                                ),
                            }
                        }
                        Some(token) if token.kind == TokenKind::Comma => {
                            let mut args = self
                                .extract_sequence_with_optional_trailing_separator_using_function(
                                    state,
                                    TokenKind::Comma,
                                    |this: &Parser, state: &mut State| {
                                        this.extract_expression(
                                            state,
                                            &[TokenKind::Rsquare, TokenKind::Comma],
                                        )
                                    },
                                    "expression",
                                    TokenKind::Comma,
                                    TokenKind::Rsquare,
                                )
                                .elems;
                            args.insert(0, expr);
                            Expression {
                                span: self.end_span(state, &expr_span_start),
                                kind: ExpressionKind::ArrayLiteral(args),
                            }
                        }
                        Some(_) => self.unexpected_token(state, format_args!("] or ; or ,")),
                    };
                }
                TokenKind::Lbrace => {
                    let block_expr = self.extract_block_expression(state);
                    break Expression {
                        span: block_expr.span,
                        kind: ExpressionKind::Block(Box::new(block_expr)),
                    };
                }
                TokenKind::KeywordReturn => {
                    self.extract_token(state);
                    stack.push(ExpressionOperatorWithLeftExpr {
                        operator_span: self.end_span(state, &expr_span_start),
                        operator: ExpressionOperator::Return,
                        left_expr: None,
                    });
                }
                TokenKind::Exclamation => {
                    self.extract_token(state);
                    stack.push(ExpressionOperatorWithLeftExpr {
                        operator_span: self.end_span(state, &expr_span_start),
                        operator: ExpressionOperator::BoolNegation,
                        left_expr: None,
                    });
                }
                TokenKind::Minus => {
                    self.extract_token(state);
                    stack.push(ExpressionOperatorWithLeftExpr {
                        operator_span: self.end_span(state, &expr_span_start),
                        operator: ExpressionOperator::ArithmeticNegation,
                        left_expr: None,
                    });
                }
                TokenKind::Ampersand => {
                    self.extract_token(state);
                    let operator = match self.peek_token(state) {
                        None => self.unexpected_eof(format_args!("mut or expression")),
                        Some(token) if token.kind == TokenKind::KeywordMut => {
                            self.extract_token(state);
                            ExpressionOperator::TakeMutReference
                        }
                        _ => ExpressionOperator::TakeReference,
                    };
                    stack.push(ExpressionOperatorWithLeftExpr {
                        operator_span: self.end_span(state, &expr_span_start),
                        operator,
                        left_expr: None,
                    });
                }
                TokenKind::Star => {
                    self.extract_token(state);
                    stack.push(ExpressionOperatorWithLeftExpr {
                        operator_span: self.end_span(state, &expr_span_start),
                        operator: ExpressionOperator::Dereference,
                        left_expr: None,
                    });
                }
                TokenKind::KeywordConst => {
                    self.extract_token(state);
                    stack.push(ExpressionOperatorWithLeftExpr {
                        operator_span: self.end_span(state, &expr_span_start),
                        operator: ExpressionOperator::Const,
                        left_expr: None,
                    });
                }
                TokenKind::KeywordIf => {
                    let if_expr = self.extract_if_expression(state);
                    break Expression {
                        span: self.end_span(state, &expr_span_start),
                        kind: ExpressionKind::If(Box::new(if_expr)),
                    };
                }
                TokenKind::KeywordLoop => {
                    self.extract_token(state);
                    let body = self.extract_block_expression(state);
                    break Expression {
                        span: self.end_span(state, &expr_span_start),
                        kind: ExpressionKind::Loop(Box::new(body)),
                    };
                }
                TokenKind::KeywordWhile => {
                    self.extract_token(state);
                    let condition = self.extract_expression(state, &[TokenKind::Rangle]);
                    let body = self.extract_block_expression(state);
                    break Expression {
                        span: self.end_span(state, &expr_span_start),
                        kind: ExpressionKind::WhileLoop {
                            condition: Box::new(condition),
                            body: Box::new(body),
                        },
                    };
                }
                _ => self.unexpected_token(state, format_args!("expression")),
            }
        }
    }

    fn extract_expression(
        &self,
        state: &mut State,
        end_token_kinds: &[TokenKind],
    ) -> Expression<'lexed> {
        let mut stack = Vec::new();
        'parser_loop: loop {
            // Extract simple expression
            let mut expr = self.expression_parser_extract_simple_expression(state, &mut stack);
            // Extract operator
            macro_rules! binary_op_impl {
                ($operator:ident, $span:expr) => {{
                    let left_expr = self.expression_parser_fold_stack(
                        &mut stack,
                        expr,
                        Some((ExpressionOperator::$operator, &$span)),
                    );
                    stack.push(ExpressionOperatorWithLeftExpr {
                        operator_span: $span.clone(),
                        operator: ExpressionOperator::$operator,
                        left_expr: Some(left_expr),
                    });
                }};
            }
            macro_rules! binary_op {
                ($operator:ident) => {{
                    let span_start = state.start_span();
                    self.extract_token(state);
                    binary_op_impl!($operator, self.end_span(state, &span_start));
                }};
            }
            // Extract operator that extends the expression
            while let Some(token) = self.peek_token(state) {
                match token.kind {
                    kind if end_token_kinds.contains(&kind) => {
                        break 'parser_loop self
                            .expression_parser_fold_stack(&mut stack, expr, None);
                    }
                    TokenKind::ColonColon => {
                        let coloncolon = self.extract_token(state);
                        match &mut expr {
                            Expression {
                                span,
                                kind: ExpressionKind::Path(path),
                            } => {
                                let ident = self.extract_identifier(state);
                                *span = self.overspan(span, &ident.span);
                                path.span = self.overspan(&path.span, &ident.span);
                                path.segments.push(PathSegment {
                                    span: ident.span,
                                    identifier: ident,
                                    generic_args: None,
                                });
                            }
                            _ => self.source_code.error_on(
                                coloncolon.slice,
                                format_args!("Unexpected :: operator, left side has to be a path"),
                                format_args!("left side is not a path"),
                                &mut [].into_iter(),
                            ),
                        }
                    }
                    TokenKind::Dot => {
                        let span_start = state.start_span();
                        self.extract_token(state);
                        let left_expr = self.expression_parser_fold_stack(
                            &mut stack,
                            expr,
                            Some((ExpressionOperator::Dot, &self.end_span(state, &span_start))),
                        );
                        let member_kind = match self.peek_token(state) {
                            None => {
                                self.unexpected_eof(format_args!("identifier or integer literal"))
                            }
                            Some(token) => match token.kind {
                                TokenKind::Identifier => {
                                    Member::Identifier(self.extract_identifier(state))
                                }
                                TokenKind::Integer => {
                                    Member::IntegerLiteral(self.extract_integer_literal(state))
                                }
                                _ => self.unexpected_token(
                                    state,
                                    format_args!("identifier or integer literal"),
                                ),
                            },
                        };
                        let member_span = match &member_kind {
                            Member::Identifier(identifier) => &identifier.span,
                            Member::IntegerLiteral(integer_literal) => &integer_literal.span,
                        };
                        expr = Expression {
                            span: self.overspan(&left_expr.span, member_span),
                            kind: ExpressionKind::MemberExpression {
                                obj: Box::new(left_expr),
                                member: member_kind,
                            },
                        };
                    }
                    TokenKind::Lparen => {
                        let span_start = state.start_span();
                        let args = self
                            .extract_sequence_with_optional_trailing_separator_using_function(
                                state,
                                TokenKind::Lparen,
                                |this: &Parser, state: &mut State| {
                                    this.extract_expression(
                                        state,
                                        &[TokenKind::Rparen, TokenKind::Comma],
                                    )
                                },
                                "expression",
                                TokenKind::Comma,
                                TokenKind::Rparen,
                            )
                            .elems;
                        let span = self.end_span(state, &span_start);
                        let left_expr = self.expression_parser_fold_stack(
                            &mut stack,
                            expr,
                            Some((ExpressionOperator::Call, &span)),
                        );
                        expr = Expression {
                            span: self.overspan(&left_expr.span, &span),
                            kind: ExpressionKind::Call {
                                callee: Box::new(left_expr),
                                args,
                            },
                        };
                    }
                    TokenKind::Langle => {
                        match &mut expr {
                            Expression {
                                span,
                                kind: ExpressionKind::Path(path),
                            } if path.segments.last().unwrap().generic_args.is_none()
                                && slices_meet(
                                    path.segments.last().unwrap().span.last().unwrap().slice,
                                    token.slice,
                                ) =>
                            {
                                let span_start = state.start_span();
                                // Generic arguments
                                let args = self
                                        .extract_sequence_with_optional_trailing_separator_using_function(
                                            state,
                                            TokenKind::Langle,
                                            |this: &Parser, state: &mut State| {
                                                this.extract_expression(
                                                    state,
                                                    &[TokenKind::Rangle, TokenKind::Comma]
                                                )
                                            },
                                            "expression",
                                            TokenKind::Comma,
                                            TokenKind::Rangle
                                        )
                                        .elems;
                                let gen_args_span = self.end_span(state, &span_start);
                                let segment = path.segments.last_mut().unwrap();
                                segment.span = self.overspan(&segment.span, &gen_args_span);
                                *span = self.overspan(span, &gen_args_span);
                                path.span = self.overspan(&path.span, &gen_args_span);
                                segment.generic_args = Some(args);
                            }
                            _ => break,
                        }
                    }
                    TokenKind::KeywordAs => {
                        let span_start = state.start_span();
                        self.extract_token(state);
                        let as_span = self.end_span(state, &span_start);
                        let left_expr = self.expression_parser_fold_stack(
                            &mut stack,
                            expr,
                            Some((ExpressionOperator::CastAs, &as_span)),
                        );
                        let r#type = self.extract_type(state);
                        expr = Expression {
                            span: self.overspan(&left_expr.span, &r#type.span),
                            kind: ExpressionKind::Cast {
                                expr: Box::new(left_expr),
                                r#type: Box::new(r#type),
                            },
                        };
                    }
                    _ => break,
                }
            }
            // Extract operator that needs right expression
            let Some(token) = self.peek_token(state) else {
                break self.expression_parser_fold_stack(&mut stack, expr, None);
            };
            match token.kind {
                kind if end_token_kinds.contains(&kind) => {
                    break self.expression_parser_fold_stack(&mut stack, expr, None);
                }
                TokenKind::Equals => binary_op!(Equal),
                TokenKind::PlusEquals => binary_op!(PlusEqual),
                TokenKind::MinusEquals => binary_op!(MinusEqual),
                TokenKind::StarEquals => binary_op!(StarEqual),
                TokenKind::SlashEquals => binary_op!(SlashEqual),
                TokenKind::PercentEquals => binary_op!(PercentEqual),
                TokenKind::PipeEquals => binary_op!(LogicOrEqual),
                TokenKind::AmpersandEquals => binary_op!(LogicAndEqual),
                TokenKind::CaretEquals => binary_op!(LogicXorEqual),
                TokenKind::LangleLangleEquals => binary_op!(LogicShiftLeftEqual),
                TokenKind::Pipe => binary_op!(LogicOr),
                TokenKind::Caret => binary_op!(LogicXor),
                TokenKind::LangleLangle => binary_op!(LogicShiftLeft),
                TokenKind::PipePipe => binary_op!(LazyBoolOr),
                TokenKind::Plus => binary_op!(Plus),
                TokenKind::Minus => binary_op!(Minus),
                TokenKind::Star => binary_op!(Star),
                TokenKind::Slash => binary_op!(Slash),
                TokenKind::Percent => binary_op!(Percent),
                TokenKind::EqualsEquals => binary_op!(EqualEqual),
                TokenKind::ExclamationEquals => binary_op!(NotEqual),
                TokenKind::LangleEquals => binary_op!(LessEqual),
                TokenKind::Langle => binary_op!(Less),
                TokenKind::Ampersand => {
                    let span_start = state.start_span();
                    let ampersand = self.extract_token(state);
                    // & or &&
                    match self.peek_token(state) {
                        Some(token)
                            if slices_meet(ampersand.slice, token.slice)
                                && token.kind == TokenKind::Ampersand =>
                        {
                            // &&
                            self.extract_token(state);
                            binary_op_impl!(LazyBoolAnd, self.end_span(state, &span_start));
                        }
                        _ => {
                            // &
                            binary_op_impl!(LogicAnd, self.end_span(state, &span_start));
                        }
                    }
                }
                TokenKind::Rangle => {
                    let span_start = state.start_span();
                    let rangle = self.extract_token(state);
                    // > or >> or >= or >>=
                    match self.peek_token(state) {
                        Some(token)
                            if slices_meet(rangle.slice, token.slice)
                                && token.kind == TokenKind::Equals =>
                        {
                            // >=
                            self.extract_token(state);
                            binary_op_impl!(GreaterEqual, self.end_span(state, &span_start));
                        }
                        Some(token)
                            if slices_meet(rangle.slice, token.slice)
                                && token.kind == TokenKind::Rangle =>
                        {
                            // >> or >>=
                            let second_rangle = self.extract_token(state);
                            match self.peek_token(state) {
                                Some(token)
                                    if slices_meet(second_rangle.slice, token.slice)
                                        && token.kind == TokenKind::Equals =>
                                {
                                    // >>=
                                    self.extract_token(state);
                                    binary_op_impl!(
                                        LogicShiftRightEqual,
                                        self.end_span(state, &span_start)
                                    );
                                }
                                _ => {
                                    // >>
                                    binary_op_impl!(
                                        LogicShiftRight,
                                        self.end_span(state, &span_start)
                                    );
                                }
                            }
                        }
                        _ => {
                            // >
                            binary_op_impl!(Greater, self.end_span(state, &span_start));
                        }
                    }
                }
                _ => break self.expression_parser_fold_stack(&mut stack, expr, None),
            }
        }
    }

    fn extract_if_expression(&self, state: &mut State) -> IfExpression<'lexed> {
        let span_start = state.start_span();
        self.extract_token_of_kind(state, TokenKind::KeywordIf);
        let condition = self.extract_expression(state, &[TokenKind::Lbrace]);
        let true_branch = self.extract_block_expression(state);
        match self.peek_token(state) {
            Some(token) if token.kind == TokenKind::KeywordElse => {
                self.extract_token(state);
                match self.peek_token(state) {
                    None => self.unexpected_eof(format_args!("if or {{")),
                    Some(token) if token.kind == TokenKind::KeywordIf => {
                        let if_expr = self.extract_if_expression(state);
                        IfExpression {
                            span: self.end_span(state, &span_start),
                            condition,
                            true_branch,
                            r#else: Some(ElseExpression::If(Box::new(if_expr))),
                        }
                    }
                    _ => {
                        let else_block = self.extract_block_expression(state);
                        IfExpression {
                            span: self.end_span(state, &span_start),
                            condition,
                            true_branch,
                            r#else: Some(ElseExpression::Block(else_block)),
                        }
                    }
                }
            }
            _ => IfExpression {
                span: self.end_span(state, &span_start),
                condition,
                true_branch,
                r#else: None,
            },
        }
    }

    fn extract_block_expression(&self, state: &mut State) -> BlockExpression<'lexed> {
        let block_span_start = state.start_span();
        self.extract_token_of_kind(state, TokenKind::Lbrace);
        let mut statements = Vec::new();
        let mut final_expression = None;
        loop {
            let Some(token) = self.peek_token(state) else {
                self.unexpected_eof(format_args!("}} or statement"));
            };
            if token.kind == TokenKind::Rbrace {
                self.extract_token(state);
                return BlockExpression {
                    span: self.end_span(state, &block_span_start),
                    statements,
                    final_expression,
                };
            }
            if let Some(expr) = final_expression.take() {
                let requires_semicolon = match expr.kind {
                    ExpressionKind::Underscore
                    | ExpressionKind::Continue
                    | ExpressionKind::Break
                    | ExpressionKind::IntegerLiteral(_)
                    | ExpressionKind::ByteLiteral(_)
                    | ExpressionKind::ByteStringLiteral(_)
                    | ExpressionKind::Path(_)
                    | ExpressionKind::Tuple(_)
                    | ExpressionKind::ArrayLiteral(_)
                    | ExpressionKind::ArrayWithSize(_, _)
                    | ExpressionKind::Return(_)
                    | ExpressionKind::BoolNegation(_)
                    | ExpressionKind::ArithmeticNegation(_)
                    | ExpressionKind::Assign(_, _)
                    | ExpressionKind::PlusAssign(_, _)
                    | ExpressionKind::MinusAssign(_, _)
                    | ExpressionKind::MultiplyAssign(_, _)
                    | ExpressionKind::DivideAssign(_, _)
                    | ExpressionKind::DivideRemainderAssign(_, _)
                    | ExpressionKind::LogicOrAssign(_, _)
                    | ExpressionKind::LogicAndAssign(_, _)
                    | ExpressionKind::LogicXorAssign(_, _)
                    | ExpressionKind::LogicShiftLeftAssign(_, _)
                    | ExpressionKind::LogicShiftRightAssign(_, _)
                    | ExpressionKind::LogicOr(_, _)
                    | ExpressionKind::LogicAnd(_, _)
                    | ExpressionKind::LogicXor(_, _)
                    | ExpressionKind::LogicShiftLeft(_, _)
                    | ExpressionKind::LogicShiftRight(_, _)
                    | ExpressionKind::Plus(_, _)
                    | ExpressionKind::Minus(_, _)
                    | ExpressionKind::Multiply(_, _)
                    | ExpressionKind::Divide(_, _)
                    | ExpressionKind::DivideRemainter(_, _)
                    | ExpressionKind::CompareEqual(_, _)
                    | ExpressionKind::CompareNotEqual(_, _)
                    | ExpressionKind::CompareGreater(_, _)
                    | ExpressionKind::CompareGreaterEqual(_, _)
                    | ExpressionKind::CompareLess(_, _)
                    | ExpressionKind::CompareLessEqual(_, _)
                    | ExpressionKind::LazyBoolAnd(_, _)
                    | ExpressionKind::LazyBoolOr(_, _)
                    | ExpressionKind::TakeReference(_)
                    | ExpressionKind::TakeMutReference(_)
                    | ExpressionKind::Dereference(_)
                    | ExpressionKind::Call { .. }
                    | ExpressionKind::Cast { .. }
                    | ExpressionKind::MemberExpression { .. }
                    | ExpressionKind::Const(_) => true,

                    ExpressionKind::Block(_)
                    | ExpressionKind::If(_)
                    | ExpressionKind::Loop(_)
                    | ExpressionKind::WhileLoop { .. } => false,
                };
                if requires_semicolon {
                    let span_start = state.start_span();
                    self.extract_token_of_kind(state, TokenKind::Semicolon);
                    statements.push(Statement {
                        span: self.overspan(&expr.span, &self.end_span(state, &span_start)),
                        kind: StatementKind::Expression(expr),
                    })
                } else {
                    statements.push(Statement {
                        span: expr.span,
                        kind: StatementKind::Expression(expr),
                    })
                }
            }

            let Some(token) = self.peek_token(state) else {
                self.unexpected_eof(format_args!("}} or statement"));
            };
            let span_start = state.start_span();
            match token.kind {
                TokenKind::Semicolon => {
                    self.extract_token(state);
                }
                TokenKind::Rbrace => {
                    self.extract_token(state);
                    return BlockExpression {
                        span: self.end_span(state, &span_start),
                        statements,
                        final_expression,
                    };
                }
                TokenKind::KeywordLet => {
                    self.extract_token(state);
                    let mutable = match self.peek_token(state) {
                        None => self.unexpected_eof(format_args!("mut or identifier")),
                        Some(token) if token.kind == TokenKind::KeywordMut => {
                            self.extract_token(state);
                            true
                        }
                        _ => false,
                    };
                    let name = self.extract_identifier(state);
                    let r#type = match self.peek_token(state) {
                        None => self.unexpected_eof(format_args!(": or =")),
                        Some(token) if token.kind == TokenKind::Colon => {
                            self.extract_token(state);
                            Some(self.extract_type(state))
                        }
                        _ => None,
                    };
                    self.extract_token_of_kind(state, TokenKind::Equals);
                    let value = self.extract_expression(state, &[TokenKind::Semicolon]);
                    self.extract_token_of_kind(state, TokenKind::Semicolon);
                    statements.push(Statement {
                        span: self.end_span(state, &span_start),
                        kind: StatementKind::Let {
                            name,
                            mutable,
                            r#type,
                            value,
                        },
                    });
                }
                TokenKind::KeywordConst
                    if self
                        .peek_token_at(state, 2)
                        .map(|token| {
                            token.kind == TokenKind::Colon || token.kind == TokenKind::Equals
                        })
                        .unwrap_or(false) =>
                {
                    self.extract_token(state);
                    let name = self.extract_identifier(state);
                    let r#type = match self.peek_token(state) {
                        None => self.unexpected_eof(format_args!(": or =")),
                        Some(token) if token.kind == TokenKind::Colon => {
                            self.extract_token(state);
                            Some(self.extract_type(state))
                        }
                        _ => None,
                    };
                    self.extract_token_of_kind(state, TokenKind::Equals);
                    let value = self.extract_expression(state, &[TokenKind::Semicolon]);
                    self.extract_token_of_kind(state, TokenKind::Semicolon);
                    statements.push(Statement {
                        span: self.end_span(state, &span_start),
                        kind: StatementKind::Const {
                            name,
                            r#type,
                            value,
                        },
                    });
                }
                _ => {
                    final_expression = Some(
                        self.extract_expression(state, &[TokenKind::Rbrace, TokenKind::Semicolon]),
                    );
                }
            }
        }
    }

    fn extract_generic_param(&self, state: &mut State) -> GenericParam<'lexed> {
        match self.peek_token(state) {
            None => self.unexpected_eof(format_args!("type or const")),
            Some(token) if token.kind == TokenKind::KeywordConst => {
                let span_start = state.start_span();
                self.extract_token(state);
                let name = self.extract_identifier(state);
                self.extract_token_of_kind(state, TokenKind::Colon);
                let r#type = self.extract_type(state);
                GenericParam {
                    span: self.end_span(state, &span_start),
                    kind: GenericParamKind::ConstValue { name, r#type },
                }
            }
            _ => {
                let type_name = self.extract_identifier(state);
                GenericParam {
                    span: type_name.span,
                    kind: GenericParamKind::Type { name: type_name },
                }
            }
        }
    }

    fn extract_generic_params(&self, state: &mut State) -> Vec<GenericParam<'lexed>> {
        self.extract_sequence_with_optional_trailing_separator_using_function(
            state,
            TokenKind::Langle,
            Self::extract_generic_param,
            "generic parameter",
            TokenKind::Comma,
            TokenKind::Rangle,
        )
        .elems
    }

    fn extract_function_param(&self, state: &mut State) -> FunctionParam<'lexed> {
        let span_start = state.start_span();
        let mutable = match self.peek_token(state) {
            None => self.unexpected_eof(format_args!("mut or identifier")),
            Some(token) if token.kind == TokenKind::KeywordMut => {
                self.extract_token(state);
                true
            }
            _ => false,
        };
        let name = self.extract_identifier(state);
        self.extract_token_of_kind(state, TokenKind::Colon);
        let r#type = self.extract_type(state);
        FunctionParam {
            span: self.end_span(state, &span_start),
            mutable,
            name,
            r#type,
        }
    }

    fn extract_function(&self, state: &mut State) -> Function<'lexed> {
        let span_start = state.start_span();
        self.extract_token_of_kind(state, TokenKind::KeywordFn);
        let name = self.extract_identifier(state);
        let generic_params = match self.peek_token(state) {
            None => self.unexpected_eof(format_args!("< or (")),
            Some(token) if token.kind == TokenKind::Langle => {
                Some(self.extract_generic_params(state))
            }
            _ => None,
        };
        let params = self
            .extract_sequence_with_optional_trailing_separator_using_function(
                state,
                TokenKind::Lparen,
                Self::extract_function_param,
                "function parameter",
                TokenKind::Comma,
                TokenKind::Rparen,
            )
            .elems;
        let return_type = match self.peek_token(state) {
            None => self.unexpected_eof(format_args!("-> or {{")),
            Some(token) if token.kind == TokenKind::MinusRangle => {
                self.extract_token(state);
                Some(self.extract_type(state))
            }
            _ => None,
        };
        let body = self.extract_block_expression(state);
        Function {
            span: self.end_span(state, &span_start),
            name,
            generic_params,
            params,
            return_type,
            body,
        }
    }

    fn extract_visibility(&self, state: &mut State) -> Visibility {
        match self.peek_token(state) {
            Some(token) if token.kind == TokenKind::KeywordPub => {
                self.extract_token(state);
                Visibility::Public
            }
            _ => Visibility::Private,
        }
    }

    fn extract_struct_field(&self, state: &mut State) -> StructField<'lexed> {
        let span_start = state.start_span();
        let visibility = self.extract_visibility(state);
        let name = self.extract_identifier(state);
        self.extract_token_of_kind(state, TokenKind::Colon);
        let r#type = self.extract_type(state);
        StructField {
            span: self.end_span(state, &span_start),
            visibility,
            name,
            r#type,
        }
    }

    fn extract_tuple_struct_field(&self, state: &mut State) -> TupleStructField<'lexed> {
        let span_start = state.start_span();
        let visibility = self.extract_visibility(state);
        let r#type = self.extract_type(state);
        TupleStructField {
            span: self.end_span(state, &span_start),
            visibility,
            r#type,
        }
    }

    fn extract_struct_or_tuple_struct(&self, state: &mut State) -> StructOrTupleStruct<'lexed> {
        let span_start = state.start_span();
        self.extract_token_of_kind(state, TokenKind::KeywordStruct);
        let name = self.extract_identifier(state);
        let generic_params = match self.peek_token(state) {
            None => self.unexpected_eof(format_args!("< or ( or {{")),
            Some(token) if token.kind == TokenKind::Langle => {
                Some(self.extract_generic_params(state))
            }
            _ => None,
        };
        match self.peek_token(state) {
            None => self.unexpected_eof(format_args!("( or {{")),
            Some(token) => match token.kind {
                TokenKind::Lparen => {
                    let field_types = self
                        .extract_sequence_with_optional_trailing_separator_using_function(
                            state,
                            TokenKind::Lparen,
                            Self::extract_tuple_struct_field,
                            "type",
                            TokenKind::Comma,
                            TokenKind::Rparen,
                        )
                        .elems;
                    self.extract_token_of_kind(state, TokenKind::Semicolon);
                    StructOrTupleStruct::TupleStruct(TupleStruct {
                        span: self.end_span(state, &span_start),
                        name,
                        generic_params,
                        field_types,
                    })
                }
                TokenKind::Lbrace => {
                    let fields = self
                        .extract_sequence_with_optional_trailing_separator_using_function(
                            state,
                            TokenKind::Lbrace,
                            Self::extract_struct_field,
                            "struct field",
                            TokenKind::Comma,
                            TokenKind::Rbrace,
                        )
                        .elems;
                    StructOrTupleStruct::Struct(Struct {
                        span: self.end_span(state, &span_start),
                        name,
                        generic_params,
                        fields,
                    })
                }
                _ => match generic_params {
                    Some(_) => self.unexpected_token(state, format_args!("( or {{")),
                    None => self.unexpected_token(state, format_args!("< or ( or {{")),
                },
            },
        }
    }

    fn extract_type_alias(&self, state: &mut State) -> TypeAlias<'lexed> {
        let span_start = state.start_span();
        self.extract_token_of_kind(state, TokenKind::KeywordType);
        let name = self.extract_identifier(state);
        let generic_params = match self.peek_token(state) {
            None => self.unexpected_eof(format_args!("< or =")),
            Some(token) if token.kind == TokenKind::Langle => {
                Some(self.extract_generic_params(state))
            }
            _ => None,
        };
        self.extract_token_of_kind(state, TokenKind::Equals);
        let r#type = self.extract_type(state);
        self.extract_token_of_kind(state, TokenKind::Semicolon);
        TypeAlias {
            span: self.end_span(state, &span_start),
            name,
            generic_params,
            r#type,
        }
    }

    fn extract_use(&self, state: &mut State) -> Use<'lexed> {
        let span_start = state.start_span();
        self.extract_token_of_kind(state, TokenKind::KeywordUse);
        let path = self.extract_path(state);
        self.extract_token_of_kind(state, TokenKind::Semicolon);
        Use {
            span: self.end_span(state, &span_start),
            path,
        }
    }

    fn extract_constant(&self, state: &mut State) -> Constant<'lexed> {
        let span_start = state.start_span();
        self.extract_token_of_kind(state, TokenKind::KeywordConst);
        let name = self.extract_identifier(state);
        let generic_params = match self.peek_token(state) {
            None => self.unexpected_eof(format_args!("< or : or =")),
            Some(token) if token.kind == TokenKind::Langle => {
                Some(self.extract_generic_params(state))
            }
            _ => None,
        };
        let r#type = match self.peek_token(state) {
            None => self.unexpected_eof(format_args!(": or =")),
            Some(token) if token.kind == TokenKind::Colon => {
                self.extract_token(state);
                Some(self.extract_type(state))
            }
            _ => None,
        };
        self.extract_token_of_kind(state, TokenKind::Equals);
        let value = self.extract_expression(state, &[TokenKind::Semicolon]);
        self.extract_token_of_kind(state, TokenKind::Semicolon);
        Constant {
            span: self.end_span(state, &span_start),
            name,
            generic_params,
            r#type,
            value,
        }
    }

    fn extract_attribute_tree(&self, state: &mut State) -> AttributeTree<'lexed> {
        let span_start = state.start_span();
        let name = self.extract_identifier(state);
        match self.peek_token(state) {
            Some(token) if token.kind == TokenKind::Lparen => {
                let ExtractedSequence { elems, .. } = self
                    .extract_sequence_with_optional_trailing_separator_using_function(
                        state,
                        TokenKind::Lparen,
                        Self::extract_attribute_tree,
                        "attribute tree",
                        TokenKind::Comma,
                        TokenKind::Rparen,
                    );
                AttributeTree {
                    span: self.end_span(state, &span_start),
                    name,
                    subtrees: Some(elems),
                }
            }
            _ => AttributeTree {
                span: name.span,
                name,
                subtrees: None,
            },
        }
    }

    fn extract_outer_attribute(&self, state: &mut State) -> OuterAttribute<'lexed> {
        let span_start = state.start_span();
        self.extract_token_of_kind(state, TokenKind::Hash);
        self.extract_token_of_kind(state, TokenKind::Lsquare);
        let attribute_tree = self.extract_attribute_tree(state);
        self.extract_token_of_kind(state, TokenKind::Rsquare);
        OuterAttribute {
            span: self.end_span(state, &span_start),
            attribute_tree,
        }
    }

    fn extract_module(&self, state: &mut State) -> Module<'lexed> {
        let span_start = state.start_span();
        self.extract_token_of_kind(state, TokenKind::KeywordMod);
        let name = self.extract_identifier(state);
        match self.peek_token(state) {
            None => self.unexpected_eof(format_args!("; or {{")),
            Some(token) => match token.kind {
                TokenKind::Semicolon => {
                    self.extract_token(state);
                    Module {
                        span: self.end_span(state, &span_start),
                        name,
                        items: None,
                    }
                }
                TokenKind::Lbrace => {
                    self.extract_token(state);
                    let items = self.extract_items(state, Some(TokenKind::Rbrace));
                    self.extract_token_of_kind(state, TokenKind::Rbrace);
                    Module {
                        span: self.end_span(state, &span_start),
                        name,
                        items: Some(items),
                    }
                }
                _ => self.unexpected_token(state, format_args!("; or {{")),
            },
        }
    }

    fn extract_item(&self, state: &mut State) -> Item<'lexed> {
        let span_start = state.start_span();
        let mut outer_attributes = Vec::new();
        let mut visibility = Visibility::Private;
        loop {
            match self.peek_token(state) {
                None => self.unexpected_eof(format_args!("item or outer attribute or pub")),
                Some(token) => match token.kind {
                    TokenKind::Hash => outer_attributes.push(self.extract_outer_attribute(state)),
                    TokenKind::KeywordPub => {
                        visibility = self.extract_visibility(state);
                        break;
                    }
                    _ => break,
                },
            }
        }
        let Some(token) = self.peek_token(state) else {
            self.unexpected_eof(format_args!("item"));
        };
        let item_kind: ItemKind = match token.kind {
            TokenKind::KeywordMod => {
                let module = self.extract_module(state);
                ItemKind::Module(module)
            }
            TokenKind::KeywordStruct => match self.extract_struct_or_tuple_struct(state) {
                StructOrTupleStruct::Struct(r#struct) => ItemKind::Struct(r#struct),
                StructOrTupleStruct::TupleStruct(tuple_struct) => {
                    ItemKind::TupleStruct(tuple_struct)
                }
            },
            TokenKind::KeywordFn => {
                let function = self.extract_function(state);
                ItemKind::Function(function)
            }
            TokenKind::KeywordUse => {
                let r#use = self.extract_use(state);
                ItemKind::Use(r#use)
            }
            TokenKind::KeywordType => {
                let type_alias = self.extract_type_alias(state);
                ItemKind::TypeAlias(type_alias)
            }
            TokenKind::KeywordConst => {
                let constant = self.extract_constant(state);
                ItemKind::Constant(constant)
            }
            _ => self.unexpected_token(state, format_args!("item")),
        };
        Item {
            span: self.end_span(state, &span_start),
            outer_attributes,
            visibility,
            kind: item_kind,
        }
    }

    fn extract_items(&self, state: &mut State, end_token: Option<TokenKind>) -> Vec<Item<'lexed>> {
        let mut items = Vec::new();
        loop {
            match self.peek_token(state) {
                token if token.map(|token| token.kind) == end_token => return items,
                _ => items.push(self.extract_item(state)),
            }
        }
    }
}

pub fn parse(lexed_program: lexer::Program) -> Program {
    Program::new(lexed_program, |lexed_program| {
        let parser = Parser {
            source_code: lexed_program.source_code(),
            lexed_program,
        };
        let mut state = State {
            current_token_idx: 0,
        };
        let items = parser.extract_items(&mut state, None);
        items
    })
}
