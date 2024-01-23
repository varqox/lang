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

mod lexer;
mod module_tree;
mod parser;
mod source_code;

fn main() -> Result<()> {
    bunt::set_stderr_color_choice(if io::stderr().is_terminal() {
        ColorChoice::Always
    } else {
        ColorChoice::Never
    });

    let args = CmdArgs::parse();

    let source_code = source_code::Program {
        code: io::read_to_string(File::open(&args.source)?)?,
        path: args.source.into(),
    };
    let lexed_program = lexer::lex(source_code);
    let parsed_program = parser::parse(lexed_program);
    let module_tree = module_tree::build(parsed_program);

    Ok(())
}
