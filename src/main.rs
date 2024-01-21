use bunt::termcolor::ColorChoice;
use clap::Parser;
use color_eyre::Result;
use std::{
    fs::File,
    io::{self, IsTerminal},
};

#[derive(Parser)]
struct CmdArgs {
    source: String,
}

mod source_code;
mod lexer;
mod parser;
mod type_checker;

fn main() -> Result<()> {
    bunt::set_stderr_color_choice(if io::stderr().is_terminal() {
        ColorChoice::Always
    } else {
        ColorChoice::Never
    });

    let args = CmdArgs::parse();
    let source_code = if args.source == "-" {
        source_code::Program {
            code: io::read_to_string(io::stdin())?,
            filename: "source".to_string(),
        }
    } else {
        source_code::Program {
            code: io::read_to_string(File::open(&args.source)?)?,
            filename: args.source,
        }
    };
    let lexed_program = lexer::lex(&source_code);
    let parsed_program = parser::parse(&lexed_program);
    dbg!(&parsed_program);

    Ok(())
}
