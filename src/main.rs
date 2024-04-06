use lazy_static::lazy_static;
use ndarray::prelude::*;
use regex::Regex;

struct BinaryArrowOperation {
    first_input_indices: Vec<String>,
    second_input_indices: Vec<String>,
    output_indices: Vec<String>,
    summation_indices: Vec<String>,
}

lazy_static! {
    static ref ARROW_EXPRESSION_REGEX: regex::Regex =
        Regex::new(r"((?:\w+ *)+), *((?:\w+ *)+)(?:->|→) *((?:\w+ *)+)").expect("valid regex");
}

fn parse_indices(indices: &str) -> Vec<String> {
    indices.split_whitespace().map(|i| i.to_owned()).collect::<Vec<_>>()
}

fn parse_arrow_expression(expression: &str) -> BinaryArrowOperation {
    let captures = ARROW_EXPRESSION_REGEX.captures(expression);
    // TODO: parse index labels out of the string
    // Collect summation indices—those that don't appear in the output.
    BinaryArrowOperation {
        first_input_indices: vec![],
        second_input_indices: vec![],
        output_indices: vec![],
        summation_indices: vec![],
    }
}

fn einsum(
    expression: &str,
    a: &Array<f64, IxDyn>,
    b: &Array<f64, IxDyn>,
) -> Result<Array<f64, IxDyn>, ()> {
    let operation = parse_arrow_expression(expression);
    // match up the index names and sizes

    // create output array with output indices

    // use recursion to simulate nested for loops for the output indices
    Err(())
}

fn main() {
    let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).into_dyn();
    let b = arr2(&[[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]]).into_dyn();

    let a_ = a.clone().into_shape((3, 3)).expect("dimensions faithful");
    let b_ = b.clone().into_shape((3, 3)).expect("dimensions faithful");

    // [[ 36.,  42.,  48.],
    //  [ 81.,  96., 111.],
    //  [126., 150., 174.]
    assert_eq!(
        einsum("i j, j k → i k", &a.clone(), &b.clone()).expect("einsummed"),
        a_.dot(&b_).into_dyn()
    );

    // [[ 2.,  6., 12.],
    //  [20., 30., 42.],
    //  [56., 72., 90.]
    assert_eq!(
        einsum("i j, i j → i j", &a.clone(), &b.clone()).expect("einsummed"),
        a * b
    );
}

#[cfg(test)]
mod tests {
    use super::{ARROW_EXPRESSION_REGEX, parse_indices};

    #[test]
    fn test_arrow_expression_regex_matches() {
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j k, l m n -> i n m"));
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j, j k -> i k"));
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j, j k → i k"));
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j, i j -> i j"));
        assert!(ARROW_EXPRESSION_REGEX.is_match("i j, i j → i j"));
    }

    #[test]
    fn test_arrow_expression_regex_groups() {
        let captures = ARROW_EXPRESSION_REGEX
            .captures("i j k, l m n -> i n m")
            .unwrap()
            .iter()
            .map(|m| m.expect("match").as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            captures,
            // TODO: could a smarter regex avoid the trailing whitespace?
            vec!["i j k, l m n -> i n m", "i j k", "l m n ", "i n m"]
        );
    }

    #[test]
    fn test_parse_indices() {
        assert_eq!(parse_indices("i j k"), vec!["i".to_owned(), "j".to_owned(), "k".to_owned()]);
        assert_eq!(parse_indices("i j k "), vec!["i".to_owned(), "j".to_owned(), "k".to_owned()])
    }
}
