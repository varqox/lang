#[derive(Debug)]
pub struct Program {
    pub code: String,
    pub filename: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocationInfo {
    line: usize,     // starts from 1
    line_pos: usize, // starts from 1
}

impl Program {
    pub fn slice_to_span(&self, slice: &str) -> std::ops::Range<usize> {
        let start = (slice.as_ptr() as usize) - (self.code.as_ptr() as usize);
        std::ops::Range {
            start,
            end: start + slice.len(),
        }
    }

    fn location_info(&self, location: usize) -> LocationInfo {
        let mut line = 1;
        let mut line_pos = 1;
        for c in self.code[..location].chars() {
            if c == '\n' {
                line += 1;
                line_pos = 1;
            } else {
                line_pos += 1;
            }
        }
        LocationInfo { line, line_pos }
    }

    fn location_info_of_last_char_before_location(&self, location: usize) -> LocationInfo {
        let mut line = 1;
        let mut line_pos = 1;
        let mut last_location = LocationInfo { line, line_pos };
        for c in self.code[..location].chars() {
            last_location = LocationInfo { line, line_pos };
            if c == '\n' {
                line += 1;
                line_pos = 1;
            } else {
                line_pos += 1;
            }
        }
        last_location
    }

    pub fn error_on(
        &self,
        slice: &str,
        error_msg: core::fmt::Arguments,
        comment_msg: core::fmt::Arguments,
    ) -> ! {
        bunt::eprintln!("{$red+bold}error{/$}{$bold}: {}{/$}", error_msg);
        let span = self.slice_to_span(slice);
        // We will show either:
        // abc
        // ^^^ comment_msg
        // where first == a, last == c
        // or
        // abc
        //  \ comment_msg, where span.is_empty() and first == last == b
        // Either way, the character at position first has to be printed.
        let first_loc_info = self.location_info(span.start);
        let last_loc_info = if span.is_empty() {
            first_loc_info.clone()
        } else {
            self.location_info_of_last_char_before_location(span.end)
        };
        let line_num_width = format!("{}", last_loc_info.line).len();

        bunt::eprintln!(
            "{:line_num_width$}{$bold+blue}-->{/$} {}:{}:{}",
            "",
            self.filename,
            first_loc_info.line,
            first_loc_info.line_pos,
            line_num_width = line_num_width
        );

        // x + 1 to start after the newline
        let beg = self.code[..span.start].rfind('\n').map_or(0, |x| x + 1);
        // end excludes newline character ending the line containing the slice last character
        let end = if !span.is_empty() && self.code.as_bytes()[span.end - 1] == b'\n' {
            // The slice ends with newline
            span.end - 1
        } else {
            self.code[span.end..]
                .find('\n')
                .unwrap_or(self.code.len() - span.end)
                + span.end
        };

        #[derive(Debug)]
        enum State {
            BeforeSpan,
            Span,
            AfterSpan,
        }
        let mut state = State::BeforeSpan;
        let mut loc_info = LocationInfo {
            line: first_loc_info.line,
            line_pos: 1,
        };
        let mut next_loc_info = LocationInfo {
            line: first_loc_info.line,
            line_pos: 1,
        };
        let mut line_comment = String::new();

        bunt::eprint!(
            "{$bold+blue}{:line_num_width$} |{/$} ",
            loc_info.line,
            line_num_width = line_num_width
        );
        for c in self.code[beg..end].chars() {
            loc_info = next_loc_info;
            bunt::eprint!("{}", c);
            if loc_info == first_loc_info {
                state = State::Span;
            }
            match state {
                State::BeforeSpan => line_comment.push(' '),
                State::Span => {
                    // Do not mark newlines in the middle of the span
                    if c != '\n' {
                        line_comment.push(if span.is_empty() { '\\' } else { '^' })
                    }
                }
                State::AfterSpan => {}
            }
            if c == '\n' {
                bunt::eprintln!(
                    "{$bold+blue}{:line_num_width$} |{/$} {[bold+red]}",
                    "",
                    line_comment,
                    line_num_width = line_num_width
                );
                line_comment = String::new();

                next_loc_info = LocationInfo {
                    line: loc_info.line + 1,
                    line_pos: 1,
                };
                bunt::eprint!(
                    "{$bold+blue}{:line_num_width$} |{/$} ",
                    next_loc_info.line,
                    line_num_width = line_num_width
                );
            } else {
                next_loc_info = LocationInfo {
                    line: loc_info.line,
                    line_pos: loc_info.line_pos + 1,
                };
            }
            if loc_info == last_loc_info {
                state = State::AfterSpan;
            }
        }

        // Add missing caret in case of span marking EOF or newline at the end
        match state {
            State::Span | State::BeforeSpan => {
                line_comment.push('^');
            }
            State::AfterSpan => {}
        }
        bunt::eprintln!(
            "\n{$bold+blue}{:line_num_width$} |{/$} {[bold+red]} {[bold+red]}",
            "",
            line_comment,
            comment_msg,
            line_num_width = line_num_width
        );

        std::process::exit(1)
    }
}
