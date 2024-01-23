use crate::{
    lexer,
    parser::{self, Visibility},
    source_code,
};
use std::{
    collections::HashMap,
    fs::File,
    io::{self, ErrorKind},
    path::PathBuf,
    rc::Rc,
};

const SOURCE_FILE_EXTENSION: &'static str = "lang";

#[derive(Debug)]
pub struct Module<'sources> {
    visibility: Visibility,
    submodules: HashMap<&'sources str, Module<'sources>>,
    items: Vec<&'sources parser::Item<'sources>>,
}

self_cell::self_cell!(
    pub struct ModuleTree {
        owner: Vec<parser::Program>,
        #[covariant]
        dependent: Module,
    }

    impl { Debug }
);

pub fn build(program: parser::Program) -> ModuleTree {
    let mut sources = vec![Rc::new(program)];

    fn process_items(
        program: &Rc<parser::Program>,
        items: &Vec<parser::Item>,
        sources: &mut Vec<Rc<parser::Program>>,
        module_dir_path: &mut PathBuf,
    ) {
        let mut modules = HashMap::<&str, parser::Span>::new();
        for item in items {
            match item.kind {
                parser::ItemKind::Module(ref module) => {
                    if let Some(prev_span) = modules.get(module.name.name) {
                        program.lexed_program().error_on(
                            module.name.span,
                            format_args!("module is already declared"),
                            format_args!("module already declared"),
                            &mut [(
                                *prev_span,
                                format_args!("previous module declaration"),
                                format_args!("previous module declaration"),
                            )]
                            .into_iter(),
                        );
                    }
                    modules.insert(module.name.name, module.name.span);
                    match module.items {
                        None => {
                            module_dir_path.push(module.name.name);
                            module_dir_path.set_extension(SOURCE_FILE_EXTENSION);

                            let file = File::open(&module_dir_path)
                                .map_err(|err| {
                                    if err.kind() == ErrorKind::NotFound {
                                        program.lexed_program().error_on(
                                            module.name.span,
                                            format_args!(
                                                "module file not found: {}",
                                                &module_dir_path.as_path().display()
                                            ),
                                            format_args!(
                                                "module file not found: {}",
                                                &module_dir_path.as_path().display()
                                            ),
                                            &mut [].into_iter(),
                                        );
                                    } else {
                                        err
                                    }
                                })
                                .unwrap();
                            let source_code = source_code::Program {
                                code: io::read_to_string(file).unwrap(),
                                path: module_dir_path.clone(),
                            };
                            let lexed_program = lexer::lex(source_code);
                            let parsed_program = parser::parse(lexed_program);

                            let program = Rc::new(parsed_program);
                            sources.push(program.clone());

                            module_dir_path.set_extension("");
                            process_items(&program, program.items(), sources, module_dir_path);
                            module_dir_path.pop();
                        }
                        Some(ref items) => {
                            module_dir_path.push(module.name.name);
                            process_items(program, items, sources, module_dir_path);
                            module_dir_path.pop();
                        }
                    }
                }
                parser::ItemKind::Struct(_)
                | parser::ItemKind::TupleStruct(_)
                | parser::ItemKind::Function(_)
                | parser::ItemKind::Use(_)
                | parser::ItemKind::TypeAlias(_)
                | parser::ItemKind::Constant(_) => {}
            }
        }
    }

    let program = sources[0].clone();
    let mut source_module_dir_path = program.lexed_program().source_code().path.clone();
    source_module_dir_path.pop();
    process_items(
        &program,
        program.items(),
        &mut sources,
        &mut source_module_dir_path,
    );

    std::mem::drop(program);
    let sources = sources
        .into_iter()
        .map(|rc| Rc::try_unwrap(rc).unwrap())
        .collect();
    ModuleTree::new(sources, |programs| {
        let path_to_program = programs
            .iter()
            .map(|program| (&program.lexed_program().source_code().path, program))
            .collect::<HashMap<_, _>>();

        fn build_module<'a>(
            program: &'a parser::Program,
            visibility: Visibility,
            items: &'a Vec<parser::Item>,
            path_to_program: &HashMap<&'a PathBuf, &'a parser::Program>,
            module_dir_path: &mut PathBuf,
        ) -> Module<'a> {
            let mut submodules = HashMap::new();
            let mut module_items = Vec::new();
            for item in items {
                match &item.kind {
                    parser::ItemKind::Module(module) => {
                        module_dir_path.push(module.name.name);
                        submodules.insert(
                            module.name.name,
                            match module.items {
                                None => {
                                    module_dir_path.set_extension(SOURCE_FILE_EXTENSION);
                                    let program = *path_to_program.get(module_dir_path).unwrap();
                                    module_dir_path.set_extension("");
                                    build_module(
                                        program,
                                        item.visibility,
                                        program.items(),
                                        path_to_program,
                                        module_dir_path,
                                    )
                                }
                                Some(ref items) => build_module(
                                    program,
                                    item.visibility,
                                    items,
                                    path_to_program,
                                    module_dir_path,
                                ),
                            },
                        );
                        module_dir_path.pop();
                    }
                    _ => module_items.push(item),
                }
            }
            Module {
                visibility,
                submodules,
                items: module_items,
            }
        }

        build_module(
            &programs[0],
            Visibility::Public,
            programs[0].items(),
            &path_to_program,
            &mut source_module_dir_path,
        )
    })
}
