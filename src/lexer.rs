use crate::source_code;
use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"[ \t\n]+|//.*\n")] // Ignore this regex pattern between tokens
pub enum TokenKind {
    #[token(".")]
    Dot,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("::")]
    ColonColon,
    #[token(";")]
    Semicolon,
    #[token("!")]
    Exclamation,
    #[token("_")]
    Underscore,
    #[token("(")]
    Lparen,
    #[token(")")]
    Rparen,
    #[token("{")]
    Lbrace,
    #[token("}")]
    Rbrace,
    #[token("[")]
    Lsquare,
    #[token("]")]
    Rsquare,
    #[token("<")]
    Langle,
    #[token(">")]
    Rangle,
    #[token("#")]
    Hash,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("|")]
    Pipe,
    #[token("&")]
    Ampersand,
    #[token("^")]
    Caret,
    #[token("||")]
    PipePipe,
    #[token("<<")]
    LangleLangle,
    #[token("=")]
    Equals,
    #[token("+=")]
    PlusEquals,
    #[token("-=")]
    MinusEquals,
    #[token("*=")]
    StarEquals,
    #[token("/=")]
    SlashEquals,
    #[token("%=")]
    PercentEquals,
    #[token("|=")]
    PipeEquals,
    #[token("&=")]
    AmpersandEquals,
    #[token("^=")]
    CaretEquals,
    #[token("<<=")]
    LangleLangleEquals,
    #[token("==")]
    EqualsEquals,
    #[token("!=")]
    ExclamationEquals,
    #[token("<=")]
    LangleEquals,
    #[token("->")]
    MinusRangle,
    #[regex("(r#)?[a-zA-Z_][a-zA-Z_0-9]*")]
    Identifier,
    #[regex("[0-9][0-9_]*[a-zA-Z_0-9]*")]
    Integer,
    #[regex(r#"b'([^'\\\n]|\\.)*'"#)]
    Byte,
    #[regex(r#"b"([^"\\\n]|\\.)*""#)]
    ByteString,
    #[token("mod")]
    KeywordMod,
    #[token("pub")]
    KeywordPub,
    #[token("struct")]
    KeywordStruct,
    #[token("type")]
    KeywordType,
    #[token("fn")]
    KeywordFn,
    #[token("const")]
    KeywordConst,
    #[token("if")]
    KeywordIf,
    #[token("else")]
    KeywordElse,
    #[token("impl")]
    KeywordImpl,
    #[token("mut")]
    KeywordMut,
    #[token("true")]
    KeywordTrue,
    #[token("false")]
    KeywordFalse,
    #[token("while")]
    KeywordWhile,
    #[token("use")]
    KeywordUse,
    #[token("let")]
    KeywordLet,
    #[token("loop")]
    KeywordLoop,
    #[token("break")]
    KeywordBreak,
    #[token("continue")]
    KeywordContinue,
    #[token("return")]
    KeywordReturn,
    #[token("as")]
    KeywordAs,
}

impl core::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Dot => ".",
            Self::Comma => ",",
            Self::Colon => ":",
            Self::ColonColon => "::",
            Self::Semicolon => ";",
            Self::Exclamation => "!",
            Self::Underscore => "_",
            Self::Lparen => "(",
            Self::Rparen => ")",
            Self::Lbrace => "{",
            Self::Rbrace => "}",
            Self::Lsquare => "[",
            Self::Rsquare => "]",
            Self::Langle => "<",
            Self::Rangle => ">",
            Self::Hash => "#",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Star => "*",
            Self::Slash => "/",
            Self::Percent => "%",
            Self::Pipe => "|",
            Self::Ampersand => "&",
            Self::Caret => "^",
            Self::PipePipe => "||",
            Self::LangleLangle => "<<",
            Self::Equals => "=",
            Self::PlusEquals => "+=",
            Self::MinusEquals => "-=",
            Self::StarEquals => "*=",
            Self::SlashEquals => "/=",
            Self::PercentEquals => "%=",
            Self::PipeEquals => "|=",
            Self::AmpersandEquals => "&=",
            Self::CaretEquals => "^=",
            Self::LangleLangleEquals => "<<=",
            Self::EqualsEquals => "==",
            Self::ExclamationEquals => "!=",
            Self::LangleEquals => "<=",
            Self::MinusRangle => "->",
            Self::Identifier => "identifier",
            Self::Integer => "number literal",
            Self::Byte => "byte literal",
            Self::ByteString => "byte string literal",
            Self::KeywordMod => "mod",
            Self::KeywordPub => "pub",
            Self::KeywordStruct => "struct",
            Self::KeywordType => "type",
            Self::KeywordFn => "fn",
            Self::KeywordConst => "const",
            Self::KeywordIf => "if",
            Self::KeywordElse => "else",
            Self::KeywordImpl => "impl",
            Self::KeywordMut => "mut",
            Self::KeywordTrue => "true",
            Self::KeywordFalse => "false",
            Self::KeywordWhile => "while",
            Self::KeywordUse => "use",
            Self::KeywordLet => "let",
            Self::KeywordLoop => "loop",
            Self::KeywordBreak => "break",
            Self::KeywordContinue => "continue",
            Self::KeywordReturn => "return",
            Self::KeywordAs => "as",
        })
    }
}

#[derive(Debug)]
pub struct Token<'code> {
    pub kind: TokenKind,
    pub slice: &'code str,
}

type Tokens<'code> = Vec<Token<'code>>;

self_cell::self_cell!(
    pub struct Program {
        owner: source_code::Program,
        #[covariant]
        dependent: Tokens,
    }

    impl {Debug}
);

impl Program {
    pub fn source_code(&self) -> &source_code::Program {
        self.borrow_owner()
    }

    pub fn tokens(&self) -> &Tokens {
        self.borrow_dependent()
    }

    pub fn error_on<'code>(
        &self,
        tokens: &[Token<'code>],
        error_msg: core::fmt::Arguments,
        comment_msg: core::fmt::Arguments,
        notes: &mut dyn Iterator<
            Item = (&[Token<'code>], core::fmt::Arguments, core::fmt::Arguments),
        >,
    ) -> ! {
        assert!(!tokens.is_empty());
        let tokens_to_source_span = |tokens: &[Token<'code>]| {
            let start = self
                .source_code()
                .slice_to_span(tokens.first().unwrap().slice)
                .start;
            let end = self
                .source_code()
                .slice_to_span(tokens.last().unwrap().slice)
                .end;
            &self.source_code().code[start..end]
        };
        self.source_code().error_on(
            tokens_to_source_span(tokens),
            error_msg,
            comment_msg,
            &mut notes.map(|(tokens, msg, comment)| (tokens_to_source_span(tokens), msg, comment)),
        );
    }
}

pub fn lex(source_code: source_code::Program) -> Program {
    Program::new(source_code, |source_code| {
        let mut lexer = TokenKind::lexer(&source_code.code);
        let mut tokens = Vec::new();
        while let Some(kind_res) = lexer.next() {
            match kind_res {
                Ok(kind) => tokens.push(Token {
                    kind,
                    slice: lexer.slice(),
                }),
                Err(_) => source_code.error_on(
                    lexer.slice(),
                    format_args!("cannot lex token"),
                    format_args!("unknown token"),
                    &mut [].into_iter(),
                ),
            }
        }
        tokens
    })
}
