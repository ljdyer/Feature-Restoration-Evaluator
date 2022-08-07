from metric_getter_helper.misc import check_same_char, display_or_print_html
from metric_getter_helper.messages import ERROR_CHARS_PER_ROW_AND_NUM_ROWS
from typing import Tuple

HTML_STYLE = """<style>
.fp{
    background-color: green
}
.fn{
    background-color: purple
}
pre {
  white-space: pre-wrap;
}
</style>"""


# ====================
def show_text_display_(ref: str,
                       hyp: str,
                       features: list,
                       feature_chars: list,
                       start_char: int = 0,
                       chars_per_row: int = None,
                       num_rows: int = None,
                       for_latex: bool = False,
                       ignore: list = None):

    if ignore is None:
        ignore = []
    chars = {'ref': list(ref), 'hyp': list(hyp)}
    labelled = label_fps_and_fns(
        chars, features, feature_chars, ignore, for_latex)
    labelled = labelled[start_char:]
    cpr_nr_given = sum([chars_per_row is not None, num_rows is not None])
    if cpr_nr_given == 1:
        raise ValueError(ERROR_CHARS_PER_ROW_AND_NUM_ROWS)
    elif cpr_nr_given == 2:
        rows = to_rows(labelled, chars_per_row, num_rows)
        if for_latex is True:
            rows = [escape_spaces_row(row) for row in rows]
            final_latex = '\n'.join(
                [f"\\texttt{{{''.join(r)}}}\\\\" for r in rows]
            )
            print(final_latex)
        else:
            html = '<br>'.join(''.join(r) for r in rows)
            final_html = HTML_STYLE + pre(html)
            display_or_print_html(final_html)
    else:
        if for_latex is True:
            final_latex = f"\\texttt{{{''.join(labelled)}}}\\\\"
            print(final_latex)
        else:
            html = HTML_STYLE + pre(''.join(labelled))
            display_or_print_html(html)


# ====================
def label_fps_and_fns(chars: dict,
                      features: list,
                      feature_chars: list,
                      ignore: list,
                      for_latex: bool = False) -> str:

    output_chars = []
    while chars['ref'] and chars['hyp']:
        next_char = {'ref': chars['ref'].pop(0), 'hyp': chars['hyp'].pop(0)}
        ignored_chars, chars = ignore_chars(chars, ignore)
        if check_same_char(next_char, chars) is not True:
            return None
        features_present, chars = get_features_present(
            next_char, chars, features)
        output_chars.extend(get_next_entries(
            next_char, features_present, features, feature_chars,
            ignored_chars=ignored_chars,
            ignore_caps='CAPITALISATION' in ignore,
            for_latex=for_latex))
    return output_chars


# ====================
def ignore_chars(chars: dict, ignore: list) -> Tuple[str, dict]:

    ignored_chars = []
    while len(chars['hyp']) > 0 and chars['hyp'][0] in ignore:
        ignored_chars.append(chars['hyp'].pop(0))
    while len(chars['ref']) > 0 and chars['ref'][0] in ignore:
        chars['ref'].pop(0)
    return ignored_chars, chars


# ====================
def get_features_present(next_char: dict, chars: dict, features: list) -> dict:

    features_present = {'ref': [], 'hyp': []}
    for ref_or_hyp in chars.keys():
        if 'CAPITALISATION' in features and next_char[ref_or_hyp].isupper():
            features_present[ref_or_hyp].append('CAPITALISATION')
        while len(chars[ref_or_hyp]) > 0 and chars[ref_or_hyp][0] in features:
            features_present[ref_or_hyp].append(chars[ref_or_hyp].pop(0))
    return features_present, chars


# ====================
def get_next_entries(next_char: dict,
                     features_present: dict,
                     features: list,
                     feature_chars: list,
                     ignored_chars: list,
                     ignore_caps: bool,
                     for_latex: bool = False) -> list:

    # print(ignored_chars)
    class_label = cmd if for_latex else span_class
    char_box = mbox if for_latex else lambda x: x
    next_entries = []
    if 'CAPITALISATION' in features:
        tfpn_ = tfpn('CAPITALISATION', features_present)
        if tfpn_ not in ['fn', 'fp'] or ignore_caps is True:
            next_entries.append(next_char['hyp'])
        else:
            next_entries.append(class_label(tfpn_, next_char['hyp']))
    else:
        next_entries.append(next_char['hyp'])
    for feature in feature_chars:
        tfpn_ = tfpn(feature, features_present)
        if tfpn_ in ['fn', 'fp']:
            next_entries.append(class_label(tfpn_, char_box(feature)))
        elif tfpn_ == 'tp' or feature in ignored_chars:
            next_entries.append(feature)
    return next_entries


# ====================
def tfpn(feature: str, features_present: dict) -> str:

    in_ref = feature in features_present['ref']
    in_hyp = feature in features_present['hyp']
    if in_hyp and in_ref:
        return 'tp'
    elif in_ref:
        return 'fn'
    elif in_hyp:
        return 'fp'
    else:
        return 'tn'


# ====================
def to_rows(entries: list, chars_per_row: int, num_rows: int) -> list:

    row_char_ranges = zip(
        range(0, chars_per_row*num_rows, chars_per_row),
        range(
            chars_per_row, chars_per_row*(num_rows+1), chars_per_row)
    )
    rows = [
        [entries[i] for i in range(a, b)]
        for (a, b) in row_char_ranges
    ]
    return rows


# ====================
def span_class(class_: str, inner_html: str) -> str:

    return f'<span class="{class_}">{inner_html}</span>'


# ====================
def pre(inner_html: str) -> str:

    return f"<pre>{inner_html}</pre>"


# ====================
def cmd(tag: str, char: str) -> str:

    return f"\\{tag}{{{char}}}"


# ====================
def mbox(char: str) -> str:

    return cmd('mbox', char)


# ====================
def fn(char: str) -> str:

    return cmd('fn', char)


# ====================
def fp(char: str) -> str:

    return cmd('fp', char)


# ====================
def escape_spaces_row(row: str) -> str:

    row = escape_row_end_spaces(row)
    row = escape_other_spaces(row)
    return row


# ====================
def escape_row_end_spaces(row: str) -> str:

    row[0] = escape_row_end_space(row[0])
    row[-1] = escape_row_end_space(row[-1])
    return row


# ====================
def escape_row_end_space(entry: str) -> str:

    if entry == ' ':
        return r"\Verb+{\ }+"
    else:
        return entry


# ====================
def escape_other_spaces(row: str) -> str:

    row = [escape_other_space(e) for e in row]
    return row


# ====================
def escape_other_space(entry: str) -> str:

    if entry == ' ':
        return r"{\ }"
    else:
        return entry
